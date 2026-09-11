#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace neurelease::model::preprocessing {

// Extracts the exact set-valued boundary, contiguous, skip, length, and repetition features used
// by offline training and native inference. The returned span remains valid until the next call.
class HashedCjFeatures {
  public:
    static constexpr std::uint32_t DefaultOrders =
        (1U << 1U) | (1U << 2U) | (1U << 3U) | (1U << 4U);
    static constexpr std::uint32_t DefaultSkips = (1U << 2U) | (1U << 3U);

    explicit HashedCjFeatures(std::uint32_t orders = DefaultOrders,
                               std::uint32_t skips = DefaultSkips);

    void configure(std::uint32_t orders, std::uint32_t skips) noexcept;
    std::span<const std::uint64_t> extract(std::span<const std::uint32_t> hanCodepoints);
    std::size_t scratchBytes() const noexcept;

    static std::uint64_t hash(std::span<const std::uint32_t> tokens) noexcept;

  private:
    void appendContiguous(std::span<const std::uint32_t> codepoints);
    void appendSkips(std::span<const std::uint32_t> codepoints);
    void appendShape(std::span<const std::uint32_t> codepoints);

    std::uint32_t orders_;
    std::uint32_t skips_;
    std::vector<std::uint64_t> hashes_;
};

} // namespace neurelease::model::preprocessing
