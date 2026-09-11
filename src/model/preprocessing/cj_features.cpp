#include "model/preprocessing/cj_features.hpp"

#include <algorithm>
#include <array>

namespace neurelease::model::preprocessing {
namespace {

constexpr std::uint64_t FnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t FnvPrime = 1099511628211ULL;
constexpr std::uint32_t Start = 0x110000;
constexpr std::uint32_t End = 0x110001;
constexpr std::uint32_t Contiguous = 0x110002;
constexpr std::uint32_t Skip2 = 0x110003;
constexpr std::uint32_t Skip3 = 0x110004;
constexpr std::uint32_t Length = 0x110005;
constexpr std::uint32_t Repeat = 0x110006;
constexpr std::uint32_t LengthBucketBase = 0x120000;

} // namespace

HashedCjFeatures::HashedCjFeatures(std::uint32_t orders, std::uint32_t skips)
    : orders_(orders), skips_(skips) {
}

void HashedCjFeatures::configure(std::uint32_t orders, std::uint32_t skips) noexcept {
    orders_ = orders;
    skips_ = skips;
}

std::span<const std::uint64_t>
HashedCjFeatures::extract(std::span<const std::uint32_t> hanCodepoints) {
    hashes_.clear();
    hashes_.reserve(hanCodepoints.size() * 6 + 16);
    appendContiguous(hanCodepoints);
    appendSkips(hanCodepoints);
    appendShape(hanCodepoints);
    std::sort(hashes_.begin(), hashes_.end());
    hashes_.erase(std::unique(hashes_.begin(), hashes_.end()), hashes_.end());
    return hashes_;
}

std::uint64_t HashedCjFeatures::hash(std::span<const std::uint32_t> tokens) noexcept {
    auto result = FnvOffset;
    const auto add = [&](std::uint32_t value) {
        for (int byte = 0; byte < 4; ++byte) {
            result ^= (value >> (8 * byte)) & 0xffU;
            result *= FnvPrime;
        }
    };
    add(static_cast<std::uint32_t>(tokens.size()));
    for (const auto token : tokens) {
        add(token);
    }
    return result;
}

void HashedCjFeatures::appendContiguous(std::span<const std::uint32_t> codepoints) {
    const auto paddedAt = [&](std::size_t index) {
        if (index == 0) {
            return Start;
        }
        if (index == codepoints.size() + 1) {
            return End;
        }
        return codepoints[index - 1];
    };

    std::array<std::uint32_t, 5> feature{};
    feature[0] = Contiguous;
    for (std::uint32_t order = 1; order <= 4; ++order) {
        if (!(orders_ & (1U << order))) {
            continue;
        }
        for (std::size_t start = 0; start + order <= codepoints.size() + 2; ++start) {
            for (std::uint32_t offset = 0; offset < order; ++offset) {
                feature[offset + 1] = paddedAt(start + offset);
            }
            hashes_.push_back(hash(std::span(feature.data(), order + 1)));
        }
    }
}

void HashedCjFeatures::appendSkips(std::span<const std::uint32_t> codepoints) {
    for (const std::uint32_t distance : {2U, 3U}) {
        if (!(skips_ & (1U << distance)) || codepoints.size() <= distance) {
            continue;
        }
        for (std::size_t start = 0; start + distance < codepoints.size(); ++start) {
            const std::array feature{
                distance == 2 ? Skip2 : Skip3,
                codepoints[start],
                codepoints[start + distance],
            };
            hashes_.push_back(hash(feature));
        }
    }
}

void HashedCjFeatures::appendShape(std::span<const std::uint32_t> codepoints) {
    const std::array lengthFeature{
        Length,
        LengthBucketBase + static_cast<std::uint32_t>(std::min<std::size_t>(codepoints.size(), 12)),
    };
    hashes_.push_back(hash(lengthFeature));

    for (std::size_t index = 0; index < codepoints.size(); ++index) {
        if (std::find(codepoints.begin(), codepoints.begin() + index, codepoints[index]) !=
            codepoints.begin() + index) {
            const std::array repeatFeature{Repeat};
            hashes_.push_back(hash(repeatFeature));
            return;
        }
    }
}

std::size_t HashedCjFeatures::scratchBytes() const noexcept {
    return hashes_.capacity() * sizeof(std::uint64_t);
}

} // namespace neurelease::model::preprocessing
