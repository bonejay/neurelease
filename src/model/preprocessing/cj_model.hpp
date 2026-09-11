#pragma once

#include "model/preprocessing/cj_features.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace neurelease::model::preprocessing {

struct CharacterModelScore {
    std::int32_t value = 0;
    std::int32_t decisionThreshold = 0;
    float scale = 1.0F;
};

// Dense power-of-two hash table for the compiled Chinese/Japanese linear model. Weights remain
// packed at 4, 8, or 16 bits in memory. Feature extraction and bucket de-duplication reuse owned
// scratch buffers, so steady-state scoring performs no allocation.
class HashedLinearCjModel {
  public:
    explicit HashedLinearCjModel(const std::filesystem::path &modelPath);

    HashedLinearCjModel(const HashedLinearCjModel &) = delete;
    HashedLinearCjModel &operator=(const HashedLinearCjModel &) = delete;
    HashedLinearCjModel(HashedLinearCjModel &&) = delete;
    HashedLinearCjModel &operator=(HashedLinearCjModel &&) = delete;

    CharacterModelScore score(std::span<const std::uint32_t> hanCodepoints);

    std::uintmax_t modelFileBytes() const noexcept;
    std::size_t runtimeTableBytes() const noexcept;
    std::size_t scratchBytes() const noexcept;
    std::uint32_t hashBuckets() const noexcept;
    std::uint32_t weightBits() const noexcept;

  private:
    void load(std::span<const std::uint8_t> serializedModel);
    std::int16_t weight(std::uint32_t bucket) const noexcept;

    std::uint32_t hashBuckets_ = 0;
    std::uint32_t weightBits_ = 0;
    std::int32_t threshold_ = 0;
    float weightScale_ = 1.0F;
    std::uintmax_t modelFileBytes_ = 0;
    std::vector<std::uint8_t> ownedModel_;
    std::span<const std::uint8_t> packedWeights_;
    std::vector<std::uint32_t> buckets_;
    HashedCjFeatures featureExtractor_;
};

} // namespace neurelease::model::preprocessing
