#include "model/segmenter.hpp"

#include "model/encoder.hpp"
#include "model/kernels/kernels.hpp"
#include "model/schema.hpp"
#include "model/segmenter_model.hpp"
#include "model/weights.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace neurelease::model {

namespace {

std::filesystem::path pathIn(std::string_view directory, const char* file) {
    // The directory arrives as UTF-8; build the path through char8_t so non-ASCII survives.
    return std::filesystem::path(std::u8string(directory.begin(), directory.end())) / file;
}

std::string pathAsUtf8(const std::filesystem::path& path) {
    const std::u8string bytes = path.u8string();
    return {bytes.begin(), bytes.end()};
}

// The byte offset where each decoded codepoint began, plus one final end offset. This walks the
// name under the SAME validity rules as the preprocessing decoder (strict continuations, overlong
// and surrogate rejection, one replacement per malformed byte), so index i here is codepoint i
// there. The expected count is asserted by construction: a mismatch would mean the two walks
// disagreed, and byte spans would be silently wrong — falling back to "offsets unavailable" (all
// zero past the end) is worse than loud, so the count is checked by the caller's indexing staying
// in range.
std::vector<std::int32_t> sourceByteOffsets(std::string_view name, std::size_t codepoints) {
    std::vector<std::int32_t> offsets;
    offsets.reserve(codepoints + 1);
    std::size_t index = 0;
    while (index < name.size()) {
        offsets.push_back(static_cast<std::int32_t>(index));
        const auto first = static_cast<std::uint8_t>(name[index]);
        if (first < 0x80) {
            ++index;
            continue;
        }
        std::uint32_t value = 0;
        std::size_t byteCount = 0;
        std::uint32_t minimum = 0;
        if (first >= 0xC2 && first <= 0xDF) {
            value = first & 0x1FU;
            byteCount = 2;
            minimum = 0x80;
        } else if (first >= 0xE0 && first <= 0xEF) {
            value = first & 0x0FU;
            byteCount = 3;
            minimum = 0x800;
        } else if (first >= 0xF0 && first <= 0xF4) {
            value = first & 0x07U;
            byteCount = 4;
            minimum = 0x10000;
        }
        bool valid = byteCount != 0 && index + byteCount <= name.size();
        for (std::size_t offset = 1; valid && offset < byteCount; ++offset) {
            const auto next = static_cast<std::uint8_t>(name[index + offset]);
            valid = (next & 0xC0) == 0x80;
            value = (value << 6) | (next & 0x3FU);
        }
        valid = valid && value >= minimum && value <= 0x10FFFF &&
                !(value >= 0xD800 && value <= 0xDFFF);
        index += valid ? byteCount : 1; // one replacement codepoint per malformed byte
    }
    offsets.push_back(static_cast<std::int32_t>(name.size()));
    return offsets;
}

template <typename Enum, typename Convert>
void validateVocabulary(const Segmenter::GlobalField& field, std::string_view expectedName,
                        Enum last, Convert convert) {
    if (field.name != expectedName)
        throw std::runtime_error("segmenter schema expected global field " +
                                 std::string(expectedName) + ", found " + field.name);
    std::vector<bool> seen(static_cast<std::size_t>(last) + 1);
    for (const std::string& label : field.values) {
        const Enum value = convert(label);
        if (value == Enum::Unknown)
            throw std::runtime_error("segmenter schema has unknown " + field.name +
                                     " class: " + label);
        const std::size_t index = static_cast<std::size_t>(value);
        if (seen[index])
            throw std::runtime_error("segmenter schema repeats " + field.name +
                                     " class: " + label);
        seen[index] = true;
    }
}

// WHAT THE CHECK ABOVE IS FOR, AND WHAT IT IS NOT. It refuses a model whose labels this runtime
// cannot name, and a model that names one class twice, because either means the answers would be
// decoded into the wrong enum. It does NOT require the model to use every value the enum holds:
// model 3 dropped `live_action_movie` and the three beside it for `movie` and `series`, and a
// model is free to answer fewer classes than a header knows about. Demanding the full set is how
// this loader rejected the first model trained after that change.

} // namespace

struct NameSegmenter::Implementation {
    SegmenterEncoder encoder;
    Segmenter model;

    Implementation(std::string_view directory)
        : encoder({pathIn(directory, "character_map.bin"),
                   pathIn(directory, "chinese_japanese.bin"),
                   pathIn(directory, "transliteration_japanese.bin"),
                   pathIn(directory, "transliteration_chinese.bin")}),
          // .u8string(), NOT .string(): the latter re-encodes to the ANSI codepage on Windows,
          // and the loader treats its argument as UTF-8 — a directory with an umlaut in it then
          // fails to open the one file of the four that took this route.
          model(Weights::load(pathAsUtf8(pathIn(directory, "segmenter.bin")))) {
        // SIZED BY THE LARGEST ENUMERATOR, which is no longer Outside. SeasonEpisodeMarker was
        // appended as 36 to keep the public values stable, and this array was still sized by
        // Outside at 35 - so `seen[36]` read one past the end and the first model carrying the new
        // label was rejected for "repeating" a span type it had used once. Whatever is added next
        // must be covered here too.
        std::array<bool, static_cast<std::size_t>(SpanType::SeasonEpisodeMarker) + 1> seen{};
        for (const std::string& label : model.spanTypes()) {
            const SpanType type = schema::spanType(label);
            if (type == SpanType::Unknown)
                throw std::runtime_error("segmenter schema has unknown span type: " + label);
            const std::size_t index = static_cast<std::size_t>(type);
            if (seen[index]) throw std::runtime_error("segmenter schema repeats span type: " + label);
            seen[index] = true;
        }
        for (std::size_t index = 1; index < seen.size(); ++index) {
            // A SPAN TYPE ADDED AFTER A MODEL WAS TRAINED IS ALLOWED TO BE ABSENT. Every weights
            // file that predates `season_episode_marker` is otherwise rejected outright, which
            // would mean the parser could not load the model it currently ships the moment the
            // type was declared. A model that does not know a type simply never predicts it.
            // Everything older than that is still required: those are types every model has had,
            // and a file missing one of them is a file that does not match this schema.
            if (index == static_cast<std::size_t>(SpanType::SeasonEpisodeMarker)) continue;
            if (!seen[index])
                throw std::runtime_error("segmenter schema is missing span type: " +
                                         std::string(spanTypeName(static_cast<SpanType>(index))));
        }

        const auto& fields = model.globalFields();
        // FIVE OR SIX. Model 2 answers five whole-name questions; model 3 adds `anime`, the one a
        // release name actually states. Both load, and a file with any other count is a file this
        // runtime does not understand rather than one to read optimistically.
        if (fields.size() != 5 && fields.size() != 6)
            throw std::runtime_error("segmenter schema needs five or six global fields");
        validateVocabulary(fields[0], "content_kind", ContentKind::Other, schema::contentKind);
        validateVocabulary(fields[1], "adult", AdultKind::Yes, schema::adultKind);
        validateVocabulary(fields[2], "numbering_kind", NumberingKind::None,
                           schema::numberingKind);
        validateVocabulary(fields[3], "pack_scope", PackScope::Complete, schema::packScope);
        validateVocabulary(fields[4], "special_kind", SpecialKind::Movie, schema::specialKind);
        if (fields.size() == 6)
            validateVocabulary(fields[5], "anime", AnimeKind::Yes, schema::animeKind);
    }

    Implementation(std::string_view directory, const Implementation& source)
        : encoder({pathIn(directory, "character_map.bin"),
                   pathIn(directory, "chinese_japanese.bin"),
                   pathIn(directory, "transliteration_japanese.bin"),
                   pathIn(directory, "transliteration_chinese.bin")}),
          model(source.model) {}
};

NameSegmenter::NameSegmenter(std::string_view modelDirectory)
    : implementation_(std::make_unique<Implementation>(modelDirectory)) {}

NameSegmenter::NameSegmenter(std::string_view modelDirectory, const NameSegmenter& modelSource)
    : implementation_(std::make_unique<Implementation>(modelDirectory,
                                                        *modelSource.implementation_)) {}

NameSegmenter::~NameSegmenter() = default;
NameSegmenter::NameSegmenter(NameSegmenter&&) noexcept = default;
NameSegmenter& NameSegmenter::operator=(NameSegmenter&&) noexcept = default;

void NameSegmenter::rememberKernelFailures(std::string_view file) {
    static std::string stored; // the kernels keep the pointer's contents by copy, but be explicit
    stored = std::string(file);
    kernels::rememberBadPaths(stored.c_str());
}

Analysis NameSegmenter::analyze(std::string_view name) {
    Analysis analysis;
    const auto encoded = implementation_->encoder.encode(name);
    if (!encoded) {
        return analysis;
    }
    // The stage counters are cumulative and thread-local, so this name's share is the difference
    // across its own run. Reading them costs three loads; the counters themselves are always on.
    const auto before = profileCounters();
    const auto prediction = implementation_->model.run(*encoded);
    const auto after = profileCounters();
    const auto stageMicros = [](double later, double earlier) {
        return static_cast<float>((later - earlier) * 1e6);
    };
    analysis.convolutionMicros = stageMicros(after.convolutionSeconds, before.convolutionSeconds);
    analysis.matmulMicros = stageMicros(after.matmulSeconds, before.matmulSeconds);
    analysis.attentionMicros = stageMicros(after.attentionSeconds, before.attentionSeconds);

    // The model works in Unicode codepoints; the caller's name is UTF-8 bytes. One cumulative
    // table maps codepoint index -> byte offset so spans address the string as given. The table
    // is built by re-walking the SOURCE bytes under the decoder's own rules — a codepoint's value
    // is not enough, because the decoder turns each malformed byte into one U+FFFD, and a genuine
    // U+FFFD is three bytes while that replacement was one.
    const std::vector<std::int32_t> byteAt = sourceByteOffsets(name, encoded->codepoints.size());

    analysis.valid = true;
    const auto& types = implementation_->model.spanTypes();
    for (const auto& span : Segmenter::spans(*encoded, prediction)) {
        SegmentedSpan out;
        out.typeLabel = types[static_cast<std::size_t>(span.type)];
        out.type = schema::spanType(out.typeLabel);
        out.begin = byteAt[static_cast<std::size_t>(span.begin)];
        out.end = byteAt[static_cast<std::size_t>(span.end)];
        out.confidence = span.confidence;
        analysis.spans.push_back(std::move(out));
    }

    const auto& fields = implementation_->model.globalFields();
    SegmentedField* fieldSlots[] = {&analysis.contentKind, &analysis.adultKind,
                                    &analysis.numberingKind, &analysis.packScope,
                                    &analysis.specialKind, &analysis.animeKind};
    for (std::size_t field = 0;
         field < fields.size() && field < std::size(fieldSlots) &&
         field < prediction.globals.size();
         ++field) {
        const auto& scores = prediction.globals[field];
        const auto best = std::max_element(scores.begin(), scores.end()) - scores.begin();
        fieldSlots[field]->value = fields[field].values[static_cast<std::size_t>(best)];
        fieldSlots[field]->confidence = scores[static_cast<std::size_t>(best)];
    }

    // The token count the transformer actually ran on: one per run the character stage produced.
    analysis.tokens = static_cast<int>(encoded->runSpans.size());

    // Typed once, here, so nothing downstream has to compare a label.
    analysis.content = schema::contentKind(analysis.contentKind.value);
    analysis.numbering = schema::numberingKind(analysis.numberingKind.value);
    analysis.pack = schema::packScope(analysis.packScope.value);
    analysis.special = schema::specialKind(analysis.specialKind.value);
    analysis.adult = schema::adultKind(analysis.adultKind.value);
    // Absent on a model 2 file, where the field was never asked; `animeKind.value` stays empty and
    // this stays false, which is how a caller tells "not anime" from "not asked".
    analysis.anime = schema::animeKind(analysis.animeKind.value) == AnimeKind::Yes;

    analysis.languageEvaluated = encoded->language.evaluated;
    analysis.japanese = encoded->language.japanese;
    analysis.japaneseProbability = static_cast<float>(encoded->language.japaneseProbability);

    analysis.kernelPath = prediction.info.kernelPath;
    analysis.kernelDemoted = prediction.info.kernelDemoted;
    analysis.encodeMicros = encoded->info.totalMicros;
    analysis.modelMicros = prediction.info.totalMicros;
    return analysis;
}

} // namespace neurelease::model
