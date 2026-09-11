#include "model/preprocessing/cj_detector.hpp"

#include <cmath>
#include <stdexcept>

namespace neurelease::model::preprocessing {
namespace {

bool isHan(std::uint32_t value) noexcept {
    return (value >= 0x3400 && value <= 0x4DBF) || (value >= 0x4E00 && value <= 0x9FFF) ||
           (value >= 0xF900 && value <= 0xFAFF) || (value >= 0x20000 && value <= 0x2FA1F);
}

bool isKana(std::uint32_t value) noexcept {
    return (value >= 0x3040 && value <= 0x309F) || (value >= 0x30A0 && value <= 0x30FF) ||
           (value >= 0x31F0 && value <= 0x31FF);
}

double logistic(double value) noexcept {
    if (value >= 0.0) {
        const auto exponential = std::exp(-value);
        return 1.0 / (1.0 + exponential);
    }
    const auto exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

} // namespace

ChineseJapaneseDetector::ChineseJapaneseDetector(const std::filesystem::path &modelPath)
    : characterModel_(modelPath) {
}

ChineseJapaneseEvidence
ChineseJapaneseDetector::evaluateOne(CjSequence cjCodepoints) {
    if (cjCodepoints.empty()) {
        throw std::invalid_argument("ChineseJapaneseDetector requires a non-empty CJ span");
    }

    bool hasKana = false;
    for (const auto codepoint : cjCodepoints) {
        if (isKana(codepoint)) {
            hasKana = true;
        } else if (!isHan(codepoint)) {
            throw std::invalid_argument(
                "ChineseJapaneseDetector accepts only Han, Hiragana, and Katakana");
        }
    }

    if (hasKana) {
        return {
            1.0,
            0.5,
            true,
        };
    }

    const auto score = characterModel_.score(cjCodepoints);
    return {
        logistic(score.value * score.scale),
        logistic(score.decisionThreshold * score.scale),
        false,
    };
}

ChineseJapaneseEvidence
ChineseJapaneseDetector::evaluate(CjSequence cjCodepoints) {
    return evaluateOne(cjCodepoints);
}

std::vector<ChineseJapaneseEvidence>
ChineseJapaneseDetector::evaluateBatch(std::span<const CjSequence> cjSequences) {
    std::vector<ChineseJapaneseEvidence> output;
    output.reserve(cjSequences.size());
    for (const auto sequence : cjSequences) {
        output.push_back(evaluateOne(sequence));
    }
    return output;
}

std::uintmax_t ChineseJapaneseDetector::modelFileBytes() const noexcept {
    return characterModel_.modelFileBytes();
}

std::size_t ChineseJapaneseDetector::runtimeTableBytes() const noexcept {
    return characterModel_.runtimeTableBytes();
}

std::size_t ChineseJapaneseDetector::scratchBytes() const noexcept {
    return characterModel_.scratchBytes();
}

} // namespace neurelease::model::preprocessing
