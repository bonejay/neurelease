#include "neurelease/release_parser.h"

#include "neurelease/parser.hpp"

#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using neurelease::ParseResult;
using neurelease::ReleaseInfo;

struct rp_result {
    ParseResult value;
};

struct rp_batch {
    std::vector<rp_result> values;
};

struct rp_parser {
    std::string modelDirectory;
    std::string error;
    std::unique_ptr<neurelease::Parser> single;
    std::unique_ptr<neurelease::BatchParser> batch;
};

namespace {

thread_local std::string globalError;

static_assert(static_cast<int>(neurelease::Field::Title) == RP_ORIGIN_TITLE);
static_assert(static_cast<int>(neurelease::Field::SpecialKind) == RP_ORIGIN_SPECIAL_KIND);
static_assert(static_cast<int>(neurelease::ContentKind::Other) == RP_CONTENT_OTHER);
static_assert(static_cast<int>(neurelease::AdultKind::Yes) == RP_ADULT_YES);
static_assert(static_cast<int>(neurelease::SpecialKind::Movie) == RP_SPECIAL_MOVIE);
static_assert(static_cast<int>(neurelease::ResolutionTier::P4320) == RP_RESOLUTION_4320P);
static_assert(static_cast<int>(neurelease::SourceKind::Cam) == RP_SOURCE_CAM);
static_assert(static_cast<int>(neurelease::VideoCodec::Vp8) == RP_CODEC_VP8);
static_assert(static_cast<int>(neurelease::MediumKind::Archive) == RP_MEDIUM_ARCHIVE);
static_assert(static_cast<int>(neurelease::SubtitleFormat::Smi) == RP_SUBTITLE_FORMAT_SMI);
static_assert(static_cast<int>(neurelease::EditionKind::Theatrical) == RP_EDITION_THEATRICAL);

void setGlobalError(std::string value) noexcept {
    try { globalError = std::move(value); }
    catch (...) { globalError.clear(); }
}

void setError(rp_parser* parser, std::string value) noexcept {
    if (parser == nullptr) {
        setGlobalError(std::move(value));
        return;
    }
    try { parser->error = std::move(value); }
    catch (...) { parser->error.clear(); }
}

rp_status exceptionStatus(rp_parser* parser, std::exception_ptr failure) noexcept {
    try {
        if (failure) std::rethrow_exception(failure);
        throw std::runtime_error("missing exception detail");
    } catch (const std::bad_alloc&) {
        setError(parser, "out of memory");
        return RP_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        setError(parser, error.what());
        return RP_ERROR_INTERNAL;
    } catch (...) {
        setError(parser, "unknown internal error");
        return RP_ERROR_INTERNAL;
    }
}

std::filesystem::path utf8Path(const char* path) {
    const std::string value(path == nullptr ? "" : path);
    return std::filesystem::path(std::u8string(value.begin(), value.end()));
}

bool modelFilesExist(const char* directory) {
    const std::filesystem::path base = utf8Path(directory);
    for (const char* file : {"segmenter.bin", "character_map.bin", "chinese_japanese.bin",
                             "transliteration_japanese.bin", "transliteration_chinese.bin"}) {
        if (!std::filesystem::is_regular_file(base / file)) return false;
    }
    return true;
}

const char* emptyAsNull(const std::string& value) noexcept {
    return value.empty() ? nullptr : value.c_str();
}

const char* labelAsNull(std::string_view value) noexcept {
    return value.empty() ? nullptr : value.data();
}

const ReleaseInfo* infoOf(const rp_result* result) noexcept {
    return result == nullptr ? nullptr : &result->value.info;
}

} // namespace

extern "C" {

uint32_t rp_abi_version(void) { return RP_ABI_VERSION; }

const char* rp_global_error_message(void) { return globalError.c_str(); }

rp_status rp_parser_new(const char* modelDirectory, rp_parser** out) {
    if (out == nullptr) return RP_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    if (modelDirectory == nullptr || *modelDirectory == '\0') {
        setGlobalError("model directory is required");
        return RP_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (!modelFilesExist(modelDirectory)) {
            setGlobalError("model directory does not contain all four required model files");
            return RP_ERROR_MODEL_NOT_FOUND;
        }
        auto parser = std::make_unique<rp_parser>();
        parser->modelDirectory = modelDirectory;
        try {
            parser->single = std::make_unique<neurelease::Parser>(parser->modelDirectory);
        } catch (const std::exception& error) {
            setGlobalError(error.what());
            return RP_ERROR_MODEL_INVALID;
        }
        *out = parser.release();
        globalError.clear();
        return RP_OK;
    } catch (...) {
        return exceptionStatus(nullptr, std::current_exception());
    }
}

void rp_parser_free(rp_parser* parser) { delete parser; }

const char* rp_last_error_message(const rp_parser* parser) {
    return parser == nullptr ? "invalid parser" : parser->error.c_str();
}

rp_status rp_set_accuracy(rp_accuracy wanted) {
    if (wanted != RP_ACCURACY_EXACT && wanted != RP_ACCURACY_FAST) {
        setGlobalError("invalid accuracy value");
        return RP_ERROR_INVALID_ARGUMENT;
    }
    neurelease::setAccuracy(wanted == RP_ACCURACY_FAST ? neurelease::Accuracy::Fast
                                                           : neurelease::Accuracy::Exact);
    return RP_OK;
}

rp_accuracy rp_get_accuracy(void) {
    return neurelease::accuracy() == neurelease::Accuracy::Fast
               ? RP_ACCURACY_FAST : RP_ACCURACY_EXACT;
}

rp_status rp_parse(rp_parser* parser, const char* name, rp_result** out) {
    if (parser == nullptr || name == nullptr || out == nullptr) {
        setError(parser, "parser, name, and output pointer are required");
        return RP_ERROR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    try {
        auto result = std::make_unique<rp_result>();
        result->value = parser->single->parse(name);
        *out = result.release();
        parser->error.clear();
        return RP_OK;
    } catch (...) {
        return exceptionStatus(parser, std::current_exception());
    }
}

rp_status rp_set_batch_threads(rp_parser* parser, int threads) {
    if (parser == nullptr || threads < 0) {
        setError(parser, "parser is required and threads must be >= 0");
        return RP_ERROR_INVALID_ARGUMENT;
    }
    try {
        // Rebuilt rather than mutated: the workers share the loaded weights through the engine,
        // so a thread-count change is a construction-time fact.
        parser->batch = std::make_unique<neurelease::BatchParser>(parser->modelDirectory,
                                                                     threads);
        parser->error.clear();
        return RP_OK;
    } catch (...) {
        return exceptionStatus(parser, std::current_exception());
    }
}

rp_status rp_parse_batch(rp_parser* parser, const char* const* names, size_t count,
                         rp_batch** out) {
    if (parser == nullptr || out == nullptr || (count != 0 && names == nullptr)) {
        setError(parser, "parser, output pointer, and non-empty input array are required");
        return RP_ERROR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    try {
        std::vector<std::string> input;
        input.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            if (names[index] == nullptr) {
                setError(parser, "batch contains a null name");
                return RP_ERROR_INVALID_ARGUMENT;
            }
            input.emplace_back(names[index]);
        }
        if (input.empty()) {
            *out = new rp_batch;
            parser->error.clear();
            return RP_OK;
        }
        if (!parser->batch)
            parser->batch = std::make_unique<neurelease::BatchParser>(parser->modelDirectory);
        std::vector<ParseResult> parsed = parser->batch->parse(input);
        auto batch = std::make_unique<rp_batch>();
        batch->values.reserve(parsed.size());
        for (ParseResult& result : parsed) batch->values.push_back({std::move(result)});
        *out = batch.release();
        parser->error.clear();
        return RP_OK;
    } catch (...) {
        return exceptionStatus(parser, std::current_exception());
    }
}

size_t rp_batch_size(const rp_batch* batch) { return batch == nullptr ? 0 : batch->values.size(); }

const rp_result* rp_batch_at(const rp_batch* batch, size_t index) {
    return batch == nullptr || index >= batch->values.size() ? nullptr : &batch->values[index];
}

void rp_batch_free(rp_batch* batch) { delete batch; }

const char* rp_str(const rp_result* result, rp_field field) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr) return nullptr;
    switch (field) {
    case RP_FIELD_TITLE: return emptyAsNull(info->title);
    case RP_FIELD_SCREEN_SIZE: return labelAsNull(neurelease::label(info->screenSize));
    case RP_FIELD_SOURCE: return labelAsNull(neurelease::label(info->source));
    case RP_FIELD_STREAMING_SERVICE: return emptyAsNull(info->streamingService);
    case RP_FIELD_VIDEO_CODEC: return labelAsNull(neurelease::label(info->videoCodec));
    case RP_FIELD_AUDIO_CODEC: return emptyAsNull(info->audioCodec);
    case RP_FIELD_AUDIO_CHANNELS: return emptyAsNull(info->audioChannels);
    case RP_FIELD_AUDIO_PROFILE: return emptyAsNull(info->audioProfile);
    case RP_FIELD_HDR: return emptyAsNull(info->hdr);
    case RP_FIELD_EDITION: return labelAsNull(neurelease::label(info->edition));
    case RP_FIELD_RELEASE_GROUP: return emptyAsNull(info->releaseGroup);
    case RP_FIELD_CONTAINER: return emptyAsNull(info->container);
    case RP_FIELD_MEDIUM: return labelAsNull(neurelease::label(info->medium));
    case RP_FIELD_SUBTITLE_FORMAT: return labelAsNull(neurelease::label(info->subtitleFormat));
    }
    return nullptr;
}

int32_t rp_int(const rp_result* result, rp_int_field field) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr) return 0;
    switch (field) {
    case RP_INT_YEAR: return info->year.value_or(0);
    case RP_INT_SEASON: return info->season.value_or(0);
    case RP_INT_SEASON_END: return info->seasonEnd.value_or(0);
    case RP_INT_EPISODE: return info->episode.value_or(0);
    case RP_INT_EPISODE_END: return info->episodeEnd.value_or(0);
    case RP_INT_ABSOLUTE_EPISODE: return info->absoluteEpisode.value_or(0);
    case RP_INT_ABSOLUTE_EPISODE_END: return info->absoluteEpisodeEnd.value_or(0);
    case RP_INT_EPISODE_COUNT: return info->episodeCount.value_or(0);
    case RP_INT_TOKENS: return result->value.analysis.tokens;
    }
    return 0;
}

int rp_flag(const rp_result* result, rp_flag_field field) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr) return 0;
    switch (field) {
    case RP_FLAG_VALID: return info->valid;
    case RP_FLAG_PACK: return info->pack;
    case RP_FLAG_SPECIALS: return info->specials;
    case RP_FLAG_COMPLETE_RANGE: return info->explicitCompleteRange;
    case RP_FLAG_REMUX: return info->remux;
    case RP_FLAG_PROPER: return info->proper;
    case RP_FLAG_REPACK: return info->repack;
    case RP_FLAG_TEN_BIT: return info->tenBit;
    case RP_FLAG_DUAL_AUDIO: return info->dualAudio;
    case RP_FLAG_MULTI_AUDIO: return info->multiAudio;
    case RP_FLAG_MULTI_SUBS: return info->multiSubs;
    case RP_FLAG_HARD_SUBS: return info->hardSubs;
    case RP_FLAG_ENGLISH_DUB: return info->englishDub;
    case RP_FLAG_LIGHT_ENCODE: return info->lightEncode;
    case RP_FLAG_AI_UPSCALE: return info->aiUpscale;
    case RP_FLAG_DEGRADED: return info->degraded;
    case RP_FLAG_ANIME: return info->anime;
    }
    return 0;
}

uint32_t rp_hdr_formats(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr) return 0;
    return (info->hdr10 ? RP_HDR_HDR10 : 0u) |
           (info->dolbyVision ? RP_HDR_DOLBY_VISION : 0u) |
           (info->hdr10Plus ? RP_HDR_HDR10_PLUS : 0u) |
           (info->hlg ? RP_HDR_HLG : 0u);
}

rp_date rp_date_value(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? rp_date{} : rp_date{info->date.year, info->date.month,
                                                 info->date.day};
}

rp_content_kind rp_content(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_CONTENT_UNKNOWN : static_cast<rp_content_kind>(info->content);
}

int rp_anime(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info != nullptr && info->anime ? 1 : 0;
}

rp_adult_kind rp_adult(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_ADULT_UNKNOWN : static_cast<rp_adult_kind>(info->adult);
}

rp_resolution_tier rp_screen_size(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_RESOLUTION_UNKNOWN
                           : static_cast<rp_resolution_tier>(info->screenSize);
}

rp_source_kind rp_source(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_SOURCE_UNKNOWN : static_cast<rp_source_kind>(info->source);
}

rp_video_codec rp_video_codec_value(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_CODEC_UNKNOWN : static_cast<rp_video_codec>(info->videoCodec);
}

rp_medium_kind rp_medium(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_MEDIUM_UNKNOWN : static_cast<rp_medium_kind>(info->medium);
}

rp_subtitle_format rp_subtitle_format_value(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_SUBTITLE_FORMAT_UNKNOWN
                           : static_cast<rp_subtitle_format>(info->subtitleFormat);
}

rp_numbering_kind rp_numbering(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_NUMBERING_UNKNOWN : static_cast<rp_numbering_kind>(info->numbering);
}

rp_pack_scope rp_pack(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_PACK_UNKNOWN : static_cast<rp_pack_scope>(info->packScope);
}

rp_special_kind rp_special(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? RP_SPECIAL_UNKNOWN : static_cast<rp_special_kind>(info->special);
}

float rp_verdict_confidence(const rp_result* result, rp_verdict which) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr) return 0.0F;
    switch (which) {
    case RP_VERDICT_CONTENT: return info->contentConfidence;
    case RP_VERDICT_ADULT: return info->adultConfidence;
    case RP_VERDICT_ANIME: return info->animeConfidence;
    case RP_VERDICT_NUMBERING: return info->numberingConfidence;
    case RP_VERDICT_PACK: return info->packScopeConfidence;
    case RP_VERDICT_SPECIAL: return info->specialConfidence;
    }
    return 0.0F;
}

size_t rp_edition_count(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? 0 : info->editions.size();
}

rp_edition_kind rp_edition_at(const rp_result* result, size_t index) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr || index >= info->editions.size()
               ? RP_EDITION_UNKNOWN : static_cast<rp_edition_kind>(info->editions[index]);
}

size_t rp_alternative_title_count(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? 0 : info->alternativeTitles.size();
}

const char* rp_alternative_title_at(const rp_result* result, size_t index) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr || index >= info->alternativeTitles.size()
               ? nullptr : info->alternativeTitles[index].c_str();
}

size_t rp_release_group_count(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? 0 : info->releaseGroups.size();
}

const char* rp_release_group_at(const rp_result* result, size_t index) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr || index >= info->releaseGroups.size() ? nullptr : info->releaseGroups[index].c_str();
}

size_t rp_language_count(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? 0 : info->languages.size();
}

const char* rp_language_at(const rp_result* result, size_t index) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr || index >= info->languages.size()
               ? nullptr : info->languages[index].c_str();
}

size_t rp_subtitle_language_count(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? 0 : info->subtitleLanguages.size();
}

const char* rp_subtitle_language_at(const rp_result* result, size_t index) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr || index >= info->subtitleLanguages.size()
               ? nullptr : info->subtitleLanguages[index].c_str();
}

size_t rp_origin_count(const rp_result* result) {
    const ReleaseInfo* info = infoOf(result);
    return info == nullptr ? 0 : info->origins.size();
}

rp_status rp_origin_at(const rp_result* result, size_t index, rp_origin_view* out) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr || out == nullptr || index >= info->origins.size())
        return RP_ERROR_INVALID_ARGUMENT;
    const neurelease::FieldOrigin& origin = info->origins[index];
    out->field = static_cast<rp_origin_field>(origin.field);
    out->value = origin.value.c_str();
    out->text = origin.text.c_str();
    out->begin = origin.begin;
    out->end = origin.end;
    out->confidence = origin.confidence;
    out->unconverted = origin.unconverted;
    return RP_OK;
}

rp_status rp_view(const rp_result* result, rp_result_view* out) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr || out == nullptr) return RP_ERROR_INVALID_ARGUMENT;
    out->title = emptyAsNull(info->title);
    out->streaming_service = emptyAsNull(info->streamingService);
    out->audio_codec = emptyAsNull(info->audioCodec);
    out->audio_channels = emptyAsNull(info->audioChannels);
    out->audio_profile = emptyAsNull(info->audioProfile);
    out->hdr = emptyAsNull(info->hdr);
    out->container = emptyAsNull(info->container);
    out->year = info->year.value_or(0);
    out->season = info->season.value_or(0);
    out->season_end = info->seasonEnd.value_or(0);
    out->episode = info->episode.value_or(0);
    out->episode_end = info->episodeEnd.value_or(0);
    out->absolute_episode = info->absoluteEpisode.value_or(0);
    out->absolute_episode_end = info->absoluteEpisodeEnd.value_or(0);
    out->episode_count = info->episodeCount.value_or(0);
    out->episode_title = emptyAsNull(info->episodeTitle);
    out->franchise_prefix = emptyAsNull(info->franchisePrefix);
    out->alternative_title_count = static_cast<uint32_t>(info->alternativeTitles.size());
    out->tokens = result->value.analysis.tokens;
    out->screen_size = static_cast<uint8_t>(info->screenSize);
    out->source = static_cast<uint8_t>(info->source);
    out->video_codec = static_cast<uint8_t>(info->videoCodec);
    out->medium = static_cast<uint8_t>(info->medium);
    out->content = static_cast<uint8_t>(info->content);
    out->numbering = static_cast<uint8_t>(info->numbering);
    out->special = static_cast<uint8_t>(info->special);
    out->adult = static_cast<uint8_t>(info->adult);
    out->pack_scope = static_cast<uint8_t>(info->packScope);
    out->subtitle_format = static_cast<uint8_t>(info->subtitleFormat);
    out->valid = info->valid ? 1 : 0;
    out->pack = info->pack ? 1 : 0;
    out->specials = info->specials ? 1 : 0;
    out->degraded = info->degraded ? 1 : 0;
    out->date = rp_date{info->date.year, info->date.month, info->date.day};
    out->content_confidence = info->contentConfidence;
    out->adult_confidence = info->adultConfidence;
    out->anime_confidence = info->animeConfidence;
    out->numbering_confidence = info->numberingConfidence;
    out->pack_confidence = info->packScopeConfidence;
    out->special_confidence = info->specialConfidence;
    out->edition_count = static_cast<uint32_t>(info->editions.size());
    out->release_group_count = static_cast<uint32_t>(info->releaseGroups.size());
    out->language_count = static_cast<uint32_t>(info->languages.size());
    out->subtitle_language_count = static_cast<uint32_t>(info->subtitleLanguages.size());
    out->origin_count = static_cast<uint32_t>(info->origins.size());
    uint32_t flags = 0;
    if (info->valid) flags |= 1U << RP_FLAG_VALID;
    if (info->pack) flags |= 1U << RP_FLAG_PACK;
    if (info->specials) flags |= 1U << RP_FLAG_SPECIALS;
    if (info->explicitCompleteRange) flags |= 1U << RP_FLAG_COMPLETE_RANGE;
    if (info->remux) flags |= 1U << RP_FLAG_REMUX;
    if (info->proper) flags |= 1U << RP_FLAG_PROPER;
    if (info->repack) flags |= 1U << RP_FLAG_REPACK;
    if (info->tenBit) flags |= 1U << RP_FLAG_TEN_BIT;
    if (info->dualAudio) flags |= 1U << RP_FLAG_DUAL_AUDIO;
    if (info->multiAudio) flags |= 1U << RP_FLAG_MULTI_AUDIO;
    if (info->multiSubs) flags |= 1U << RP_FLAG_MULTI_SUBS;
    if (info->hardSubs) flags |= 1U << RP_FLAG_HARD_SUBS;
    if (info->englishDub) flags |= 1U << RP_FLAG_ENGLISH_DUB;
    if (info->lightEncode) flags |= 1U << RP_FLAG_LIGHT_ENCODE;
    if (info->aiUpscale) flags |= 1U << RP_FLAG_AI_UPSCALE;
    if (info->degraded) flags |= 1U << RP_FLAG_DEGRADED;
    if (info->anime) flags |= 1U << RP_FLAG_ANIME;
    out->flags = flags;
    out->hdr_formats = rp_hdr_formats(result);
    // STATED IS THE OPTIONAL'S ENGAGEMENT, nothing else: the mapper only assigns a number it read
    // from a span, so a written S00 is engaged with 0 and an absent season is disengaged.
    uint32_t stated = 0;
    if (info->year) stated |= 1U;
    if (info->season) stated |= 2U;
    if (info->seasonEnd) stated |= 4U;
    if (info->episode) stated |= 8U;
    if (info->episodeEnd) stated |= 16U;
    if (info->absoluteEpisode) stated |= 32U;
    if (info->absoluteEpisodeEnd) stated |= 64U;
    if (info->episodeCount) stated |= 128U;
    // A zero is stated when the evidence carries the span the value came from: S00 and E00 are
    // real, and only the origins can tell a written zero from an absent field.
    //
    // AN ORIGIN WITH NO VALUE STATES NOTHING. The mapper records a marker span even when its text
    // names no number - the model marked `Staffel` and left the `2` in the title, or marked the
    // bare word `Season` - and that origin was engaging the stated bit with an unset optional
    // behind it, so `Die Simpsons 2. Staffel Folge 5` reported season 0. A written S00 records
    // the string "0" and still qualifies; a marker that states no number now reads as absent,
    // which is what it is. The underlying miss is the segmenter's and only training fixes it;
    // this stops it being reported as a confident wrong answer.
    for (const neurelease::FieldOrigin& origin : info->origins) {
        if (origin.value.empty()) continue;
        switch (origin.field) {
        case neurelease::Field::Year: stated |= 1U; break;
        case neurelease::Field::Season: stated |= 2U; break;
        case neurelease::Field::Episode: stated |= 8U; break;
        case neurelease::Field::AbsoluteEpisode: stated |= 32U; break;
        default: break;
        }
    }
    out->stated = stated;
    return RP_OK;
}

size_t rp_origins_fill(const rp_result* result, rp_origin_view* out, size_t capacity) {
    const ReleaseInfo* info = infoOf(result);
    if (info == nullptr || out == nullptr) return 0;
    const size_t count = std::min(capacity, info->origins.size());
    for (size_t index = 0; index < count; ++index) {
        const neurelease::FieldOrigin& origin = info->origins[index];
        out[index].field = static_cast<rp_origin_field>(origin.field);
        out[index].value = origin.value.c_str();
        out[index].text = origin.text.c_str();
        out[index].begin = origin.begin;
        out[index].end = origin.end;
        out[index].confidence = origin.confidence;
        out[index].unconverted = origin.unconverted;
    }
    return count;
}

void rp_result_free(rp_result* result) { delete result; }

} // extern "C"
