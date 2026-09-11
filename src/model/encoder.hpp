// Turns one release name into the segmenter's input: the three field arrays, the pseudo-token
// map, and the run table.
//
// This is the C++ statement of what training saw. The field rule mirrors pretraining/fields.py
// (`one_position`, checked against the vectorised `convert` there), and the tokenisation mirrors
// finetuning/segments.py `runs` plus the gap+run token map in finetuning/two_level.py `batch`.
// tests/segmenter_model_tests.cpp holds the two sides together: it compares this encoder's output
// byte-for-byte against fields the Python pipeline dumped for real names. Change either side
// without the other and that test fails — which is the point.
//
// The heavy lifting — UTF-8, script routing, the Chinese/Japanese decision, transliteration —
// is CharacterPreprocessor, the same code the training corpus was built with.

#pragma once

#include "model/preprocessing/character_preprocessor.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace neurelease::model {

// Where the encoding time went, one name at a time. Same philosophy as RunDiagnostics on the
// model side: nobody needs it to use the result, but a caller that logs or profiles gets the
// full story from the return value — including the routing phase, which contains the
// Chinese/Japanese SVM language decision.
struct EncodeDiagnostics {
    float routeMicros = 0.0F;         // script routing, INCLUDING the CJ language decision
    float transliterateMicros = 0.0F; // token emission: transliteration, sentinels, BOS/EOS
    float totalMicros = 0.0F;         // whole encode; the rest is decode, fields and runs
};

// One name, ready for Segmenter::run. Positions are MODEL positions (after transliteration
// expansion, BOS and EOS included); run spans are half-open over the decoded source codepoints,
// which is what maps a prediction back onto the untouched name.
struct EncodedName {
    std::vector<std::int32_t> chars;
    std::vector<std::int32_t> script;
    std::vector<std::int32_t> caseFlags;
    // Which pseudo-token each model position belongs to; tokenCount marks "no token" (the BOS,
    // the EOS, and everything after the last run).
    std::vector<std::int32_t> tokenOf;
    std::int32_t tokenCount = 0;
    // Per run: the model position of its first character. The split head reads the character
    // stage here; its length is the number of runs the model actually sees.
    std::vector<std::int32_t> firstPosition;
    // Per run seen by the model: [begin, end) in source codepoints.
    std::vector<std::pair<std::int32_t, std::int32_t>> runSpans;
    std::vector<std::uint32_t> codepoints;
    // The filename-global Chinese/Japanese decision the routing made (evaluated only when the
    // name carries Han or kana): which language, with what probability, over how many codepoints.
    preprocessing::CjDecision language;
    EncodeDiagnostics info;
};

// The three preprocessing tables. Production passes real paths; an empty contextual path falls
// back to per-character readings, which is NOT what the model was trained on — accept that only
// where Chinese and Japanese names do not matter.
struct EncoderModels {
    std::filesystem::path characterMap;
    std::filesystem::path chineseJapanese;
    std::filesystem::path japaneseContextual;
    std::filesystem::path chineseContextual;
};

class SegmenterEncoder {
    public:
    explicit SegmenterEncoder(const EncoderModels &models);

    // Empty when the name cannot be encoded (invalid or overlong input). Not thread-safe: the
    // preprocessor reuses scratch buffers, so give each thread its own encoder.
    [[nodiscard]] std::optional<EncodedName> encode(std::string_view utf8);

    private:
    preprocessing::CharacterPreprocessor preprocessor_;
    std::vector<std::uint32_t> codepoints_;
    std::vector<preprocessing::Route> routes_;
    std::vector<preprocessing::TokenBag> tokens_;
};

} // namespace neurelease::model
