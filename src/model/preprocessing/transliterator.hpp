#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace neurelease::model::preprocessing {

// One fixed-position result from a contextual longest-match transliteration.
//
// Letters are zero-based a-z values and remain unordered within this source position. Match
// coordinates identify the complete source phrase which supplied the position's bag; they are
// diagnostic metadata and do not add model positions.
struct ContextualPosition {
    std::array<std::uint8_t, 12> letters{};
    std::uint8_t letterCount = 0;
    std::uint8_t matchOffset = 0;
    std::uint8_t matchLength = 0;

    bool matched() const noexcept {
        return matchLength != 0;
    }
};

// Loads an offline-compiled phrase trie and emits one Latin-letter bag per source codepoint.
//
// Matching is greedy longest-prefix with a maximum source length of twelve. The binary already
// contains stable per-position divisions for irregular readings, so runtime work is only sorted
// edge lookup and copying small letter bags. The class owns immutable tables and allocates no
// memory after the caller's reusable output vector reaches its maximum span.
class ContextualTransliterator {
    public:
    explicit ContextualTransliterator(const std::filesystem::path &modelPath);

    void encode(std::span<const std::uint32_t> codepoints,
                std::vector<ContextualPosition> &output) const;

    std::uintmax_t modelFileBytes() const noexcept;
    std::size_t runtimeTableBytes() const noexcept;

    private:
    struct Node {
        std::uint32_t edgeOffset = 0;
        std::uint32_t terminal = 0;
        std::uint16_t edgeCount = 0;
    };

    struct Edge {
        std::uint32_t codepoint = 0;
        std::uint32_t child = 0;
    };

    struct Terminal {
        std::uint32_t bagOffset = 0;
        std::uint16_t bagCount = 0;
    };

    struct Bag {
        std::uint32_t letterOffset = 0;
        std::uint16_t letterCount = 0;
    };

    std::uint32_t child(std::uint32_t node, std::uint32_t codepoint) const noexcept;
    void copyMatch(std::uint32_t terminal,
                   std::size_t sourceStart,
                   std::size_t sourceLength,
                   std::vector<ContextualPosition> &output) const;

    std::uintmax_t modelFileBytes_ = 0;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<Terminal> terminals_;
    std::vector<std::uint32_t> terminalBags_;
    std::vector<Bag> bags_;
    std::vector<std::uint8_t> letters_;
};

} // namespace neurelease::model::preprocessing
