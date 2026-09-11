#include "model/encoder.hpp"

#include "text/unicode.hpp"

#include <algorithm>
#include <chrono>

// A consumer may define `emit` as a macro. It has no meaning in this implementation.
#ifdef emit
#undef emit
#endif

namespace neurelease::model {

namespace text = neurelease::text;

namespace {

// The embedding vocabulary rows and field ids, as pretraining/fields.py defines them. Duplicated
// numbers, not duplicated logic: the values are frozen by every trained checkpoint in existence,
// so restating them here is no more a liability than the weights file is. The reference test
// compares full encodings against the Python pipeline, which is what actually pins these.
constexpr std::int32_t MaskRow = 1;
constexpr std::int32_t FirstAsciiRow = 2, LastAsciiRow = 96;
constexpr std::int32_t FirstSentinelRow = 129, LastSentinelRow = 135;
constexpr std::int32_t UpperRow = 136;
constexpr std::int32_t UnknownRow = 137, BosRow = 138, EosRow = 139;
constexpr std::int32_t FirstOriginRow = 123, LastOriginRow = 128; // HAN JA KO CYR MOD LIG

enum Script : std::int32_t {
    ScriptAscii = 1, ScriptSentinel = 2, ScriptUnknown = 3, ScriptMask = 4,
    ScriptBos = 5, ScriptEos = 6,
    // START_JA=7, JA=8, START_HAN=9, HAN=10, START_KO=11, KO=12, START_CYR=13, CYR=14,
    // START_MOD=15, MOD=16, START_LIG=17, LIG=18 — computed from the origin row below.
};

enum Case : std::int32_t { CaseNa = 1, CaseLower = 2, CaseUpper = 3 };

// Origin row -> the CONTINUATION script id; the start id is one less. Rows 123..128 map to
// (START_HAN=9, HAN=10), (START_JA=7, JA=8), (START_KO=11, KO=12), (START_CYR=13, CYR=14),
// (START_MOD=15, MOD=16), (START_LIG=17, LIG=18).
constexpr std::int32_t OriginScript[6] = {10, 8, 12, 14, 16, 18};

struct FieldsAtPosition {
    std::int32_t chars = UnknownRow;
    std::int32_t script = ScriptAscii;
    std::int32_t caseFlag = CaseNa;
};

// pretraining/fields.py `one_position`, one bag at a time.
FieldsAtPosition fieldsFor(const preprocessing::TokenBag &bag, bool isStart) {
    FieldsAtPosition fields;

    std::int32_t character = -1;
    std::int32_t sentinel = -1;
    std::int32_t origin = -1;
    bool upper = false;
    for (std::uint8_t index = 0; index < bag.size; ++index) {
        const std::int32_t row = bag.ids[index];
        if (row == BosRow || row == EosRow || row == MaskRow) {
            fields.chars = row;
            fields.script = row == BosRow ? ScriptBos : (row == EosRow ? ScriptEos : ScriptMask);
            return fields;
        }
        if (character < 0 && row >= FirstAsciiRow && row <= LastAsciiRow) {
            character = row;
        }
        if (sentinel < 0 && row >= FirstSentinelRow && row <= LastSentinelRow) {
            sentinel = row;
        }
        if (origin < 0 && row >= FirstOriginRow && row <= LastOriginRow) {
            origin = row;
        }
        upper = upper || row == UpperRow;
    }

    fields.chars = character >= 0 ? character : (sentinel >= 0 ? sentinel : UnknownRow);
    if (origin >= 0) {
        const std::int32_t continuation = OriginScript[origin - FirstOriginRow];
        fields.script = isStart ? continuation - 1 : continuation;
    } else if (fields.chars >= FirstSentinelRow && fields.chars <= LastSentinelRow) {
        fields.script = ScriptSentinel;
    } else if (fields.chars == UnknownRow) {
        fields.script = ScriptUnknown;
    } else {
        fields.script = ScriptAscii;
    }

    // ASCII rows sit at codepoint - 30, so lowercase letters span 'a'-30 .. 'z'-30. The UPPER
    // marker only means something on a letter; digits and punctuation stay NA.
    const bool letter = fields.chars >= 'a' - 30 && fields.chars <= 'z' - 30;
    fields.caseFlag = letter ? (upper ? CaseUpper : CaseLower) : CaseNa;
    return fields;
}

// finetuning/segments.py RUN, restated over codepoints. A run is a maximal stretch of decimal
// digits, or of letters — where `!?。！？～〜` and the emoji variation selector count as letters,
// because measured on labels they sit inside spans, never between them.
bool isRunDigit(char32_t codepoint) noexcept {
    return text::isDecimalDigit(codepoint);
}

bool isRunLetter(char32_t codepoint) noexcept {
    switch (codepoint) {
    case U'!':
    case U'?':
    case 0x3002: // 。
    case 0xFF01: // ！
    case 0xFF1F: // ？
    case 0xFF5E: // ～
    case 0x301C: // 〜
    case 0xFE0F: // variation selector-16; the labels treat it as part of its emoji
        return true;
    default:
        break;
    }
    // Lu/Ll/Lt/Lm/Lo plus Nl and No - Python's [^\W\d_] keeps the letter-numbers and other
    // numbers, splitting only decimal digits out - via the generated UCD tables.
    return text::isRunLetterCategory(codepoint);
}

std::vector<std::pair<std::int32_t, std::int32_t>>
runSpans(const std::vector<std::uint32_t> &codepoints) {
    std::vector<std::pair<std::int32_t, std::int32_t>> spans;
    const auto size = static_cast<std::int32_t>(codepoints.size());
    std::int32_t at = 0;
    while (at < size) {
        const char32_t codepoint = codepoints[static_cast<std::size_t>(at)];
        const bool digit = isRunDigit(codepoint);
        if (!digit && !isRunLetter(codepoint)) {
            ++at;
            continue;
        }
        const std::int32_t begin = at;
        ++at;
        while (at < size) {
            const char32_t next = codepoints[static_cast<std::size_t>(at)];
            if (digit ? !isRunDigit(next) : (!isRunLetter(next) || isRunDigit(next))) {
                break;
            }
            ++at;
        }
        spans.emplace_back(begin, at);
    }
    return spans;
}

} // namespace

SegmenterEncoder::SegmenterEncoder(const EncoderModels &models)
    : preprocessor_(models.characterMap, models.chineseJapanese, models.japaneseContextual,
                    models.chineseContextual) {}

std::optional<EncodedName> SegmenterEncoder::encode(std::string_view utf8) {
    using Clock = std::chrono::steady_clock;
    const auto micros = [](Clock::time_point from, Clock::time_point to) {
        return std::chrono::duration<float, std::micro>(to - from).count();
    };
    const auto started = Clock::now();

    EncodedName encoded;
    if (!preprocessor_.decode(utf8, codepoints_)) {
        return std::nullopt;
    }
    const auto decoded = Clock::now();
    preprocessor_.route(codepoints_, routes_, nullptr, &encoded.language);
    const auto routed = Clock::now();
    preprocessor_.emit(codepoints_, routes_, tokens_);
    const auto emitted = Clock::now();
    const auto expanded = preprocessor_.expand(tokens_);

    encoded.codepoints = codepoints_;
    const auto positions = expanded.tokens.size();
    encoded.chars.resize(positions);
    encoded.script.resize(positions);
    encoded.caseFlags.resize(positions);

    // A position starts a new source character when its source index differs from its
    // predecessor's; the character stage's boundary between them is where labels live.
    std::vector<std::int32_t> startPositions;
    startPositions.reserve(codepoints_.size() + 2);
    for (std::size_t position = 0; position < positions; ++position) {
        const bool isStart = position == 0 || expanded.sourcePositions[position] !=
                                                  expanded.sourcePositions[position - 1];
        if (isStart) {
            startPositions.push_back(static_cast<std::int32_t>(position));
        }
        const auto fields = fieldsFor(expanded.tokens[position], isStart);
        encoded.chars[position] = fields.chars;
        encoded.script[position] = fields.script;
        encoded.caseFlags[position] = fields.caseFlag;
    }
    // Drop the BOS and EOS starts: what remains is the model position of each source character,
    // exactly the `starts_at` array the training batches were built from.
    if (startPositions.size() < 2) {
        return std::nullopt;
    }
    startPositions.erase(startPositions.begin());
    startPositions.pop_back();

    const auto spans = runSpans(codepoints_);
    encoded.tokenCount = std::max<std::int32_t>(static_cast<std::int32_t>(spans.size()), 1);
    encoded.tokenOf.assign(positions, encoded.tokenCount);

    // GAP + RUN: a token owns the separators that precede it, so "I follow a bracket" survives
    // the max-pool. Positions before the first run belong to nobody.
    const auto characters = static_cast<std::int32_t>(startPositions.size());
    std::int32_t previousEnd = 0;
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const auto [begin, end] = spans[index];
        if (begin >= characters) {
            break;
        }
        const std::int32_t last = std::min(end - 1, characters - 1);
        const std::int32_t low =
            previousEnd < characters ? startPositions[static_cast<std::size_t>(previousEnd)] : 0;
        const std::int32_t high = startPositions[static_cast<std::size_t>(last)];
        for (std::int32_t position = low; position <= high; ++position) {
            encoded.tokenOf[static_cast<std::size_t>(position)] =
                static_cast<std::int32_t>(index);
        }
        encoded.firstPosition.push_back(startPositions[static_cast<std::size_t>(begin)]);
        encoded.runSpans.emplace_back(begin, end);
        previousEnd = end;
    }

    const auto finished = Clock::now();
    encoded.info.routeMicros = micros(decoded, routed);
    encoded.info.transliterateMicros = micros(routed, emitted);
    encoded.info.totalMicros = micros(started, finished);
    return encoded;
}

} // namespace neurelease::model
