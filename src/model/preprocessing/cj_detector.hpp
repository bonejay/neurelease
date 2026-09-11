#pragma once

#include "model/preprocessing/cj_model.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace neurelease::model::preprocessing {

struct ChineseJapaneseEvidence {
    // Han-only values are the model's logistic score, not proof of a third-language absence.
    // Calibration must be measured before adding an application-specific uncertainty band.
    double japaneseProbability = 0.0;
    double japaneseDecisionThreshold = 0.5;
    bool deterministic = false;
};

using CjSequence = std::span<const std::uint32_t>;

// Estimates which reading family should be used for one Chinese/Japanese sequence.
//
// The caller may concatenate separated Han/Kana runs from one filename before evaluation. This
// class accepts only Han, Hiragana, and Katakana. Kana is deterministic Japanese evidence;
// Han-only input is scored by a small compiled character-feature model. Probability and threshold
// remain separate so policy belongs to the caller. 
class ChineseJapaneseDetector {
    public:
    explicit ChineseJapaneseDetector(const std::filesystem::path &modelPath);

    ChineseJapaneseEvidence evaluate(CjSequence cjCodepoints);

    std::vector<ChineseJapaneseEvidence> evaluateBatch(std::span<const CjSequence> cjSequences);

    std::uintmax_t modelFileBytes() const noexcept;
    std::size_t runtimeTableBytes() const noexcept;
    std::size_t scratchBytes() const noexcept;

    private:
    ChineseJapaneseEvidence evaluateOne(CjSequence cjCodepoints);
    HashedLinearCjModel characterModel_;
};

} // namespace neurelease::model::preprocessing
