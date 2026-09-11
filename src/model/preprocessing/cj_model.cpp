#include "model/preprocessing/cj_model.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace neurelease::model::preprocessing {
namespace {

template <typename T> T readLittleEndian(std::span<const std::uint8_t> input,
                                         std::size_t &offset) {
    static_assert(std::is_integral_v<T>);
    if (input.size() - std::min(input.size(), offset) < sizeof(T)) {
        throw std::runtime_error("truncated CJ linear model");
    }
    using Unsigned = std::make_unsigned_t<T>;
    Unsigned result = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        result |= static_cast<Unsigned>(input[offset++]) << (index * 8U);
    }
    return std::bit_cast<T>(result);
}

float readFloat(std::span<const std::uint8_t> input, std::size_t &offset) {
    return std::bit_cast<float>(readLittleEndian<std::uint32_t>(input, offset));
}

std::size_t packedBytes(std::uint32_t buckets, std::uint32_t bits) {
    return (static_cast<std::size_t>(buckets) * bits + 7U) / 8U;
}

} // namespace

HashedLinearCjModel::HashedLinearCjModel(const std::filesystem::path &modelPath) {
    std::ifstream input(modelPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open CJ linear model " + modelPath.string());
    }
    ownedModel_.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (input.bad()) {
        throw std::runtime_error("cannot read CJ linear model " + modelPath.string());
    }
    load(ownedModel_);
}

void HashedLinearCjModel::load(std::span<const std::uint8_t> serializedModel) {
    constexpr std::size_t headerBytes = 36;
    if (serializedModel.size() < headerBytes ||
        std::memcmp(serializedModel.data(), "CJHSM1\0\0", 8) != 0) {
        throw std::runtime_error("invalid CJ linear model magic");
    }
    std::size_t offset = 8;
    const auto version = readLittleEndian<std::uint32_t>(serializedModel, offset);
    hashBuckets_ = readLittleEndian<std::uint32_t>(serializedModel, offset);
    const auto orders = readLittleEndian<std::uint32_t>(serializedModel, offset);
    const auto skips = readLittleEndian<std::uint32_t>(serializedModel, offset);
    weightBits_ = readLittleEndian<std::uint32_t>(serializedModel, offset);
    threshold_ = readLittleEndian<std::int32_t>(serializedModel, offset);
    weightScale_ = readFloat(serializedModel, offset);
    if (version != 1 || hashBuckets_ == 0 || (hashBuckets_ & (hashBuckets_ - 1U)) != 0 ||
        (weightBits_ != 4 && weightBits_ != 8 && weightBits_ != 16)) {
        throw std::runtime_error("unsupported CJ linear model header");
    }
    featureExtractor_.configure(orders, skips);
    const auto weightsBytes = packedBytes(hashBuckets_, weightBits_);
    if (serializedModel.size() != headerBytes + weightsBytes) {
        throw std::runtime_error("CJ linear model payload has the wrong size");
    }
    packedWeights_ = serializedModel.subspan(headerBytes, weightsBytes);
    modelFileBytes_ = serializedModel.size();
}

CharacterModelScore HashedLinearCjModel::score(std::span<const std::uint32_t> hanCodepoints) {
    const auto hashes = featureExtractor_.extract(hanCodepoints);
    buckets_.clear();
    buckets_.reserve(hashes.size());
    const auto mask = hashBuckets_ - 1U;
    for (const auto hash : hashes) {
        buckets_.push_back(static_cast<std::uint32_t>(hash) & mask);
    }
    std::sort(buckets_.begin(), buckets_.end());
    buckets_.erase(std::unique(buckets_.begin(), buckets_.end()), buckets_.end());

    std::int32_t value = 0;
    for (const auto bucket : buckets_) {
        value += weight(bucket);
    }
    return {value, threshold_, weightScale_};
}

std::int16_t HashedLinearCjModel::weight(std::uint32_t bucket) const noexcept {
    if (weightBits_ == 16) {
        const auto offset = static_cast<std::size_t>(bucket) * 2U;
        const auto bits = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(packedWeights_[offset]) |
            (static_cast<std::uint16_t>(packedWeights_[offset + 1]) << 8U));
        return std::bit_cast<std::int16_t>(bits);
    }
    if (weightBits_ == 8) {
        return std::bit_cast<std::int8_t>(packedWeights_[bucket]);
    }
    const auto byte = packedWeights_[bucket / 2U];
    const auto nibble = static_cast<std::uint8_t>(
        bucket % 2U == 0 ? byte & 0x0fU : (byte >> 4U) & 0x0fU);
    return static_cast<std::int16_t>(nibble < 8U ? nibble : static_cast<int>(nibble) - 16);
}

std::uintmax_t HashedLinearCjModel::modelFileBytes() const noexcept {
    return modelFileBytes_;
}

std::size_t HashedLinearCjModel::runtimeTableBytes() const noexcept {
    return packedWeights_.size();
}

std::size_t HashedLinearCjModel::scratchBytes() const noexcept {
    return featureExtractor_.scratchBytes() + buckets_.capacity() * sizeof(std::uint32_t);
}

std::uint32_t HashedLinearCjModel::hashBuckets() const noexcept {
    return hashBuckets_;
}

std::uint32_t HashedLinearCjModel::weightBits() const noexcept {
    return weightBits_;
}

} // namespace neurelease::model::preprocessing
