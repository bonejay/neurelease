#include "model/preprocessing/character_preprocessor.hpp"

#include "model/preprocessing/character_vocabulary.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace neurelease::model::preprocessing {
namespace {
namespace table = neurelease::model::preprocessing::vocabulary;

enum class Script : std::uint8_t {
    Ascii,
    ModifiedLatin,
    Cyrillic,
    Hangul,
    Han,
    Hiragana,
    Katakana,
    Other,
};

std::size_t decodeUtf8(std::string_view input, std::vector<std::uint32_t> &output) {
    output.clear();
    output.reserve(input.size());
    std::size_t invalid = 0;

    for (std::size_t index = 0; index < input.size();) {
        const auto first = static_cast<std::uint8_t>(input[index]);
        if (first < 0x80) {
            output.push_back(first);
            ++index;
            continue;
        }

        std::uint32_t value = 0;
        std::size_t byteCount = 0;
        std::uint32_t minimum = 0;
        if (first >= 0xC2 && first <= 0xDF) {
            value = first & 0x1F;
            byteCount = 2;
            minimum = 0x80;
        } else if (first >= 0xE0 && first <= 0xEF) {
            value = first & 0x0F;
            byteCount = 3;
            minimum = 0x800;
        } else if (first >= 0xF0 && first <= 0xF4) {
            value = first & 0x07;
            byteCount = 4;
            minimum = 0x10000;
        }

        bool valid = byteCount && index + byteCount <= input.size();
        for (std::size_t offset = 1; valid && offset < byteCount; ++offset) {
            const auto next = static_cast<std::uint8_t>(input[index + offset]);
            valid = (next & 0xC0) == 0x80;
            value = (value << 6) | (next & 0x3F);
        }
        valid =
            valid && value >= minimum && value <= 0x10FFFF && !(value >= 0xD800 && value <= 0xDFFF);

        if (!valid) {
            output.push_back(0xFFFD);
            ++invalid;
            ++index;
        } else {
            output.push_back(value);
            index += byteCount;
        }
    }
    return invalid;
}

bool isHan(std::uint32_t value) noexcept {
    return (value >= 0x3400 && value <= 0x4DBF) || (value >= 0x4E00 && value <= 0x9FFF) ||
           (value >= 0xF900 && value <= 0xFAFF) || (value >= 0x20000 && value <= 0x2FA1F);
}

bool isHiragana(std::uint32_t value) noexcept {
    return value >= 0x3040 && value <= 0x309F;
}

bool isKatakana(std::uint32_t value) noexcept {
    return (value >= 0x30A0 && value <= 0x30FF) || (value >= 0x31F0 && value <= 0x31FF);
}

bool isHangul(std::uint32_t value) noexcept {
    return (value >= 0x1100 && value <= 0x11FF) || (value >= 0xAC00 && value <= 0xD7AF);
}

bool isCyrillic(std::uint32_t value) noexcept {
    return value >= 0x0400 && value <= 0x052F;
}

bool isModifiedLatinRange(std::uint32_t value) noexcept {
    if (value == 0x00D7 || value == 0x00F7) {
        return false; // multiplication and division signs are symbols, not letters
    }
    return (value >= 0x00C0 && value <= 0x024F) || (value >= 0x1D00 && value <= 0x1D7F) ||
           (value >= 0x1E00 && value <= 0x1EFF) || (value >= 0x2C60 && value <= 0x2C7F) ||
           (value >= 0xA720 && value <= 0xA7FF);
}

bool isCj(Script script) noexcept {
    return script == Script::Han || script == Script::Hiragana || script == Script::Katakana;
}

std::uint32_t foldCompatibility(std::uint32_t value) noexcept {
    if (value >= 0xFF01 && value <= 0xFF5E) {
        return value - 0xFEE0;
    }
    return value == 0x3000 ? 0x20 : value;
}

Script scriptFor(std::uint32_t source, const CharacterTables &tables) noexcept {
    const auto value = foldCompatibility(source);
    if (value <= 0x7F) {
        return Script::Ascii;
    }
    if (isHan(value)) {
        return Script::Han;
    }
    if (isHiragana(value)) {
        return Script::Hiragana;
    }
    if (isKatakana(value)) {
        return Script::Katakana;
    }
    if (isHangul(value)) {
        return Script::Hangul;
    }
    if (isCyrillic(value)) {
        return Script::Cyrillic;
    }
    if (tables.latinModified.find(value) || isModifiedLatinRange(value)) {
        return Script::ModifiedLatin;
    }
    return Script::Other;
}

bool isUppercaseSource(std::uint32_t value) noexcept {
    if (value >= 'A' && value <= 'Z') {
        return true;
    }
    if ((value >= 0x00C0 && value <= 0x00D6) || (value >= 0x00D8 && value <= 0x00DE)) {
        return true;
    }
    if ((value >= 0x0410 && value <= 0x042F) || value == 0x0401) {
        return true;
    }
    return false;
}

std::uint16_t sentinelFor(std::uint32_t value) noexcept {
    if (value == 0xFFFD || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
        return table::kUnknownCharacter;
    }
    if (value == 0x00A0 || value == 0x1680 || (value >= 0x2000 && value <= 0x200A) ||
        value == 0x2028 || value == 0x2029 || value == 0x202F || value == 0x205F) {
        return 134;
    }
    if ((value >= 0x0300 && value <= 0x036F) || (value >= 0x1AB0 && value <= 0x1AFF) ||
        (value >= 0x20D0 && value <= 0x20FF)) {
        return 131;
    }
    if ((value >= 0x2000 && value <= 0x206F) || (value >= 0x3000 && value <= 0x303F)) {
        return 133;
    }
    if ((value >= 0x1F000 && value <= 0x1FAFF) || (value >= 0x20A0 && value <= 0x20CF) ||
        (value >= 0x2100 && value <= 0x27FF)) {
        return 132;
    }
    if ((value >= 0x0660 && value <= 0x0669) || (value >= 0x06F0 && value <= 0x06F9) ||
        (value >= 0xFF10 && value <= 0xFF19)) {
        return 130;
    }
    return 129;
}

bool unpackReading(std::optional<std::uint64_t> reading, TokenBag &bag) {
    if (!reading) return false;
    const auto count = static_cast<std::uint8_t>(*reading & 0xF);
    for (std::uint8_t index = 0; index < count; ++index) {
        const auto letter = (*reading >> (4 + 5 * index)) & 0x1F;
        // Packed readings store a=1 through z=26. Map them back onto the ordinary printable-ASCII
        // rows so generated readings and authored romaji share the same learned embeddings. The
        // separate script-origin row tells the model that these letters came from transliteration.
        constexpr auto LowercaseA = static_cast<std::uint16_t>('a' - 30);
        bag.add(static_cast<std::uint16_t>(LowercaseA + letter - 1));
    }
    return count != 0;
}

void unpackContextualReading(const ContextualPosition &reading, TokenBag &bag) {
    constexpr auto LowercaseA = static_cast<std::uint16_t>('a' - 30);
    for (std::uint8_t index = 0; index < reading.letterCount; ++index) {
        bag.add(static_cast<std::uint16_t>(LowercaseA + reading.letters[index]));
    }
}

void unpackLatin(std::uint64_t packed, TokenBag &bag) {
    const auto count = static_cast<std::uint8_t>(packed & 0xF);
    for (std::uint8_t index = 0; index < count; ++index) {
        auto identifier = static_cast<std::uint16_t>((packed >> (4 + 7 * index)) & 0x7F);
        constexpr auto UppercaseA = static_cast<std::uint16_t>('A' - 30);
        constexpr auto UppercaseZ = static_cast<std::uint16_t>('Z' - 30);
        if (identifier >= UppercaseA && identifier <= UppercaseZ) {
            identifier += 'a' - 'A';
        }
        bag.add(identifier);
    }
}

} // namespace

const char *routeName(Route route) noexcept {
    constexpr std::array names{
        "ascii",
        "modified_latin",
        "cyrillic",
        "hangul",
        "chinese_han",
        "japanese_cj",
        "sentinel",
        "unknown",
    };
    return names[static_cast<std::size_t>(route)];
}

CharacterPreprocessor::CharacterPreprocessor(const std::filesystem::path &characterMap,
                                             const std::filesystem::path &chineseJapaneseModel,
                                             const std::filesystem::path &japaneseContextualModel,
                                             const std::filesystem::path &chineseContextualModel)
    : tables_(CharacterTables::load(characterMap)), chineseJapaneseDetector_(chineseJapaneseModel) {
    if (!japaneseContextualModel.empty()) {
        japaneseContextual_.emplace(japaneseContextualModel);
    }
    if (!chineseContextualModel.empty()) {
        chineseContextual_.emplace(chineseContextualModel);
    }
}

bool CharacterPreprocessor::decode(std::string_view utf8,
                                   std::vector<std::uint32_t> &codepoints,
                                   Counters *counters) const {
    if (utf8.size() > MaximumNameBytes) {
        if (counters) {
            ++counters->overlongNames;
        }
        codepoints.clear();
        return false;
    }
    // AN EMBEDDED NUL IS NOT A NAME. No release name contains one; a caller that hands one over
    // has passed a buffer, not a string, and the bytes after it are not ours to read. The 2026-09-07
    // weights found a title in `" abc"` where the earlier ones did not, which is a change in the
    // model and not a reason to parse garbage - so it is refused here, before the model sees it.
    if (utf8.find(' ') != std::string_view::npos) {
        codepoints.clear();
        return false;
    }
    const auto invalid = decodeUtf8(utf8, codepoints);
    if (counters) {
        counters->invalidUtf8 += invalid;
    }
    // THE MULTIPLICATION SIGN IS AN `x`. Uploaders write `2x7` and `2×7` for the same thing,
    // and only the first parsed: the sign is not a letter, so the run splitter treated it as a
    // separator and the numbering lost its middle. Folded here, before the runs are cut, because
    // that is the decision it has to survive - the later `foldCompatibility` runs after them. One
    // codepoint for one codepoint, so every span offset still points where it did, and `x` is a
    // character the model already reads, so no retraining is implied.
    for (auto &value : codepoints) {
        if (value == 0x00D7) {
            value = 'x';
        }
    }
    return true;
}

void CharacterPreprocessor::route(std::span<const std::uint32_t> codepoints,
                                  std::vector<Route> &routes,
                                  Counters *counters,
                                  CjDecision *decision) {
    routes.resize(codepoints.size());
    cjScratch_.clear();
    cjPositions_.clear();
    if (decision) {
        *decision = {};
    }

    // One filename supplies one Chinese/Japanese decision. Short Han runs are often ambiguous by
    // themselves, while another CJ run in the same name usually provides decisive context.
    // ASCII and punctuation are skipped rather than becoming classifier features.
    for (std::size_t position = 0; position < codepoints.size(); ++position) {
        const auto script = scriptFor(codepoints[position], tables_);
        if (isCj(script)) {
            // Remembered rather than re-derived: the decision below has to paint these positions,
            // and script classification is the most expensive part of this loop.
            cjScratch_.push_back(codepoints[position]);
            cjPositions_.push_back(static_cast<std::uint32_t>(position));
            continue;
        }

        switch (script) {
        case Script::Ascii:
            routes[position] = Route::Ascii;
            break;
        case Script::ModifiedLatin:
            routes[position] = Route::ModifiedLatin;
            break;
        case Script::Cyrillic:
            routes[position] = Route::Cyrillic;
            break;
        case Script::Hangul:
            routes[position] = Route::Hangul;
            break;
        default:
            routes[position] = Route::Sentinel;
            break;
        }
    }

    if (cjScratch_.empty()) {
        return;
    }

    const auto evidence = chineseJapaneseDetector_.evaluate(cjScratch_);
    const bool japanese = evidence.japaneseProbability >= evidence.japaneseDecisionThreshold;
    if (!evidence.deterministic && counters) {
        ++counters->chineseJapaneseCalls;
    }
    if (decision) {
        decision->evaluated = true;
        decision->deterministic = evidence.deterministic;
        decision->japanese = japanese;
        decision->japaneseProbability = evidence.japaneseProbability;
        decision->japaneseDecisionThreshold = evidence.japaneseDecisionThreshold;
        for (const auto value : cjScratch_) {
            if (isHiragana(value) || isKatakana(value)) {
                ++decision->kanaCodepoints;
            } else {
                ++decision->hanCodepoints;
            }
        }
    }

    const auto cjRoute = japanese ? Route::JapaneseCj : Route::ChineseHan;
    for (const auto position : cjPositions_) {
        routes[position] = cjRoute;
    }
}

void CharacterPreprocessor::emit(std::span<const std::uint32_t> codepoints,
                                 std::span<const Route> routes,
                                 std::vector<TokenBag> &output,
                                 Counters *counters,
                                 std::vector<ContextualTrace> *contextualTrace) const {
    output.clear();
    appendTokens(codepoints, routes, output, counters, contextualTrace);
}

void CharacterPreprocessor::appendTokens(std::span<const std::uint32_t> codepoints,
                                         std::span<const Route> routes,
                                         std::vector<TokenBag> &output,
                                         Counters *counters,
                                         std::vector<ContextualTrace> *contextualTrace) const {
    // A batch reserves its whole flat buffer up front, so this is a no-op there. Reserving once
    // also lets the loop below hold a reference into the vector without risking a reallocation.
    output.reserve(output.size() + codepoints.size() + 2);
    if (contextualTrace) {
        contextualTrace->assign(codepoints.size(), {});
    }

    // Contextual readings are decoded once per contiguous Japanese or Chinese span. Keeping only
    // the active span avoids allocating and copying a second full-name position array.
    std::size_t contextualStart = 0;
    std::size_t contextualEnd = 0;

    output.emplace_back().add(table::kBeginName);

    for (std::size_t position = 0; position < codepoints.size(); ++position) {
        const auto source = codepoints[position];
        const auto value = foldCompatibility(source);
        auto route = routes[position];
        // Filled in place. A stack bag copied in afterwards costs a second write of all 34 bytes
        // per source position, which is most of this loop's memory traffic.
        TokenBag &bag = output.emplace_back();
        bool fallback = false;

        if (position >= contextualEnd) {
            contextualStart = position;
            contextualEnd = position + 1;
            contextualSpanScratch_.clear();

            const ContextualTransliterator *transliterator = nullptr;
            if (route == Route::JapaneseCj && japaneseContextual_) {
                transliterator = &*japaneseContextual_;
            } else if (route == Route::ChineseHan && chineseContextual_) {
                transliterator = &*chineseContextual_;
            }

            if (transliterator) {
                while (contextualEnd < codepoints.size() && routes[contextualEnd] == route) {
                    ++contextualEnd;
                }
                transliterator->encode(
                    codepoints.subspan(contextualStart, contextualEnd - contextualStart),
                    contextualSpanScratch_);
            }
        }

        const ContextualPosition *contextual = nullptr;
        if (!contextualSpanScratch_.empty()) {
            contextual = &contextualSpanScratch_[position - contextualStart];
            if (contextual->matched()) {
                if (counters) {
                    ++counters->contextualPositions;
                    counters->contextualMatches += contextual->matchOffset == 0;
                }
                if (contextualTrace) {
                    (*contextualTrace)[position] = {
                        static_cast<std::uint32_t>(position - contextual->matchOffset),
                        contextual->matchLength,
                        contextual->matchOffset,
                    };
                }
            } else if (counters) {
                ++counters->contextualFallbackPositions;
            }
        }

        if (route == Route::Ascii && value >= 32 && value <= 126) {
            const auto lowercase = value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
            bag.add(static_cast<std::uint16_t>(lowercase - 30));
        } else if (route == Route::Ascii) {
            if (value == 0x20) {
                bag.add(134);
            } else if (value < 32 || value == 127) {
                bag.add(135);
            } else {
                bag.add(133);
            }
        } else if (route == Route::ModifiedLatin) {
            if (const auto reading = tables_.latinModified.find(value)) {
                unpackLatin(*reading, bag);
                const bool ligature =
                    value == 0x00C6 || value == 0x00E6 || value == 0x0152 || value == 0x0153;
                bag.add(ligature ? table::kLatinLigature : table::kLatinModified);
            } else {
                bag.add(table::kUnknownCharacter);
                bag.add(table::kLatinModified);
                fallback = true;
                route = Route::Unknown;
            }
        } else if (route == Route::ChineseHan) {
            if (contextual && contextual->matched()) {
                unpackContextualReading(*contextual, bag);
            } else {
                fallback = !unpackReading(tables_.hanGeneric.find(value), bag);
            }
            if (fallback) {
                bag.add(table::kUnknownCharacter);
            }
            bag.add(table::kOriginHanGeneric);
        } else if (route == Route::JapaneseCj) {
            if (contextual && contextual->matched()) {
                unpackContextualReading(*contextual, bag);
            } else {
                std::optional<std::uint64_t> reading = isHan(value)
                    ? tables_.hanJapanese.find(value)
                    : tables_.kanaJapanese.find(value);
                if (!reading && isHan(value)) {
                    reading = tables_.hanGeneric.find(value);
                }
                fallback = !unpackReading(reading, bag);
            }
            if (fallback) {
                bag.add(table::kUnknownCharacter);
            }
            bag.add(table::kOriginJapanese);
        } else if (route == Route::Hangul) {
            fallback = !unpackReading(tables_.hangul.find(value), bag);
            if (fallback) {
                bag.add(table::kUnknownCharacter);
            }
            bag.add(table::kOriginKorean);
        } else if (route == Route::Cyrillic) {
            fallback = !unpackReading(tables_.cyrillic.find(value), bag);
            if (fallback) {
                bag.add(table::kUnknownCharacter);
            }
            bag.add(table::kOriginCyrillic);
        } else {
            const auto sentinel = sentinelFor(value);
            bag.add(sentinel);
            if (sentinel == table::kUnknownCharacter) {
                route = Route::Unknown;
            }
        }

        if (isUppercaseSource(source) &&
            (route == Route::Ascii || route == Route::Cyrillic || route == Route::ModifiedLatin)) {
            bag.add(table::kSourceUppercase);
        }

        if (counters) {
            counters->componentIds += bag.size;
            counters->routes[static_cast<std::size_t>(route)]++;
            counters->lookupFallbacks += fallback;
        }
    }

    output.emplace_back().add(table::kEndName);

    if (counters) {
        counters->componentIds += 2;
        counters->sourcePositions += codepoints.size();
        // The row's own positions, not the buffer's: a batch appends every row into one buffer.
        counters->outputPositions += codepoints.size() + 2;
    }
}

// Appends one name's positions. A rejected name contributes none, which is the empty row a batch
// wants and the empty result a single call wants.
bool CharacterPreprocessor::processOne(std::string_view utf8,
                                       std::vector<TokenBag> &output,
                                       Counters *counters) {
    if (counters) {
        ++counters->names;
        counters->sourceBytes += utf8.size();
    }
    if (!decode(utf8, codepoints_, counters)) {
        return false;
    }
    route(codepoints_, routes_, counters);
    appendTokens(codepoints_, routes_, output, counters, nullptr);
    return true;
}

PreprocessedName CharacterPreprocessor::process(std::string_view utf8, Counters &counters) {
    PreprocessedName result;
    result.valid = processOne(utf8, result.tokens, &counters);
    return result;
}

PreprocessedName CharacterPreprocessor::process(std::string_view utf8) {
    PreprocessedName result;
    result.valid = processOne(utf8, result.tokens, nullptr);
    return result;
}

PreprocessedBatch CharacterPreprocessor::processBatchImpl(std::span<const std::string> names,
                                                          Counters *counters) {
    PreprocessedBatch result;
    result.offsets.reserve(names.size() + 1);
    result.valid.reserve(names.size());
    result.offsets.push_back(0);

    // UTF-8 bytes bound the codepoints they encode, so this is an upper bound and exact for the
    // ASCII names that dominate. Reserving it once means the row emits below never reallocate and
    // every row lands in its final position — no per-row staging buffer, no second copy.
    std::size_t maximumTokenCount = names.size() * 2;
    for (const auto &name : names) {
        maximumTokenCount += name.size();
    }
    result.tokens.reserve(maximumTokenCount);

    for (const auto &name : names) {
        const auto success = processOne(name, result.tokens, counters);
        result.valid.push_back(static_cast<std::uint8_t>(success));
        result.offsets.push_back(result.tokens.size());
    }
    return result;
}

PreprocessedBatch CharacterPreprocessor::processBatch(std::span<const std::string> names) {
    return processBatchImpl(names, nullptr);
}

PreprocessedBatch CharacterPreprocessor::processBatch(std::span<const std::string> names,
                                                      Counters &counters) {
    return processBatchImpl(names, &counters);
}

ExpandedName CharacterPreprocessor::expand(std::span<const TokenBag> compact) const {
    // Lowercase letters occupy one contiguous run of rows; everything the encoder adds alongside
    // them — script origin, uppercase, ligature — sits at or above the first marker row.
    constexpr auto FirstLetter = static_cast<std::uint16_t>('a' - 30);
    constexpr auto LastLetter = static_cast<std::uint16_t>('z' - 30);

    ExpandedName result;
    result.tokens.reserve(compact.size() * 2);
    result.sourcePositions.reserve(compact.size() * 2);

    for (std::size_t position = 0; position < compact.size(); ++position) {
        const auto &bag = compact[position];
        std::size_t letters = 0;
        for (std::uint8_t index = 0; index < bag.size; ++index) {
            letters += bag.ids[index] >= FirstLetter && bag.ids[index] <= LastLetter;
        }

        if (letters < 2) {
            // One letter or none: already one position per letter, and nothing to mark.
            result.tokens.push_back(bag);
            result.sourcePositions.push_back(static_cast<std::uint32_t>(position));
            continue;
        }

        std::size_t emitted = 0;
        for (std::uint8_t index = 0; index < bag.size; ++index) {
            const auto row = bag.ids[index];
            if (row < FirstLetter || row > LastLetter) {
                continue;
            }
            TokenBag &expanded = result.tokens.emplace_back();
            expanded.add(row);
            if (emitted == 0) {
                expanded.add(CharacterStartRow);
            }
            // Every position keeps the markers, so "this came from Japanese" holds across the
            // whole reading rather than only where the character happened to begin.
            for (std::uint8_t other = 0; other < bag.size; ++other) {
                const auto marker = bag.ids[other];
                if (marker < FirstLetter || marker > LastLetter) {
                    expanded.add(marker);
                }
            }
            result.sourcePositions.push_back(static_cast<std::uint32_t>(position));
            ++emitted;
        }
    }
    return result;
}

PreparedName CharacterPreprocessor::prepare(std::string_view utf8) {
    PreparedName result;
    if (!decode(utf8, result.codepoints)) {
        throw std::runtime_error("sample name exceeds preprocessing limit");
    }
    route(result.codepoints, result.routes, nullptr, &result.cj);
    return result;
}

bool CharacterPreprocessor::containsNonLatinScript(std::string_view utf8) {
    if (!decode(utf8, codepoints_)) {
        return false;
    }
    return std::any_of(codepoints_.begin(), codepoints_.end(), [this](std::uint32_t codepoint) {
        const auto script = scriptFor(codepoint, tables_);
        return script != Script::Ascii && script != Script::ModifiedLatin;
    });
}

ChineseJapaneseDetector &CharacterPreprocessor::chineseJapaneseDetector() noexcept {
    return chineseJapaneseDetector_;
}

const ChineseJapaneseDetector &CharacterPreprocessor::chineseJapaneseDetector() const noexcept {
    return chineseJapaneseDetector_;
}

std::size_t CharacterPreprocessor::reusableScratchBytes() const noexcept {
    return codepoints_.capacity() * sizeof(std::uint32_t) + routes_.capacity() * sizeof(Route) +
           cjScratch_.capacity() * sizeof(std::uint32_t) +
           cjPositions_.capacity() * sizeof(std::uint32_t) +
           contextualSpanScratch_.capacity() * sizeof(ContextualPosition) +
           chineseJapaneseDetector_.scratchBytes();
}

std::uintmax_t CharacterPreprocessor::contextualModelFileBytes() const noexcept {
    return (japaneseContextual_ ? japaneseContextual_->modelFileBytes() : 0) +
           (chineseContextual_ ? chineseContextual_->modelFileBytes() : 0);
}

std::size_t CharacterPreprocessor::contextualRuntimeTableBytes() const noexcept {
    return (japaneseContextual_ ? japaneseContextual_->runtimeTableBytes() : 0) +
           (chineseContextual_ ? chineseContextual_->runtimeTableBytes() : 0);
}

std::size_t CharacterPreprocessor::characterTableBytes() const noexcept {
    return tables_.bytes();
}

} // namespace neurelease::model::preprocessing
