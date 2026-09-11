#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace neurelease::model::preprocessing {

inline constexpr std::size_t MaximumNameBytes = 16 * 1024;

enum class Route : std::uint8_t {
    Ascii,
    ModifiedLatin,
    Cyrillic,
    Hangul,
    ChineseHan,
    JapaneseCj,
    Sentinel,
    Unknown,
};

inline constexpr std::size_t RouteCount = 8;

const char *routeName(Route route) noexcept;

struct TokenBag {
    // Multiple evidence IDs share one source position and will be pooled by the character model.
    //
    // DELIBERATELY UNINITIALIZED. A position holds one or two IDs on average, so zeroing all 16
    // slots wrote 32 bytes to store four, once per source codepoint — the single largest piece of
    // memory traffic in emission. `size` already says how far the bag is filled, so the slots past
    // it are simply never read. That only holds while every reader honours `size`, which is why
    // equality is written out below instead of defaulted: a defaulted comparison would compare the
    // leftover slots too and report two identical bags as different.
    std::array<std::uint16_t, 16> ids;
    std::uint8_t size = 0;

    // User-provided on purpose. Without it the default constructor is implicit, and
    // `vector::emplace_back()` value-initializes — which zero-fills the whole object first and
    // reinstates exactly the wipe the uninitialized `ids` above exists to avoid.
    TokenBag() noexcept {}

    void add(std::uint16_t identifier) {
        if (size == ids.size()) {
            throw std::runtime_error("token bag overflow");
        }
        ids[size++] = identifier;
    }

    bool contains(std::uint16_t identifier) const noexcept {
        return std::find(ids.begin(), ids.begin() + size, identifier) != ids.begin() + size;
    }

    bool operator==(const TokenBag &other) const noexcept {
        return size == other.size && std::equal(ids.begin(), ids.begin() + size, other.ids.begin());
    }
};

struct Counters {
    // Optional observability for tests and benchmarks; normal preprocessing does not need it.
    std::uint64_t names = 0;
    std::uint64_t sourceBytes = 0;
    std::uint64_t sourcePositions = 0;
    std::uint64_t outputPositions = 0;
    std::uint64_t componentIds = 0;
    std::uint64_t invalidUtf8 = 0;
    std::uint64_t lookupFallbacks = 0;
    std::uint64_t chineseJapaneseCalls = 0;
    std::uint64_t contextualMatches = 0;
    std::uint64_t contextualPositions = 0;
    std::uint64_t contextualFallbackPositions = 0;
    std::uint64_t overlongNames = 0;
    std::uint64_t malformedCsvRows = 0;
    std::array<std::uint64_t, RouteCount> routes{};
};

// Rows the preprocessor owns rather than the generated tables.
//
// Sweeping the whole corpus showed 86 of the 150 vocabulary rows are ever emitted, and 140 onward
// are reserved and untouched. `CharacterStartRow` marks the first position of an expanded source
// character, which is all that is needed to recover character boundaries: every position without
// it continues the one before. A four-way start/inside/end/single scheme would say the same thing
// with three more rows.
inline constexpr std::uint16_t CharacterStartRow = 140;

// Expanded output alongside the source position each entry came from.
//
// The compact form keeps one position per source codepoint, which is what makes a predicted span
// trivially mappable back to the untouched name. Expanding 進 into s-h-i-n breaks that one-to-one
// relation, so the mapping has to be carried explicitly.
struct ExpandedName {
    std::vector<TokenBag> tokens;
    std::vector<std::uint32_t> sourcePositions;
};

// Half-open span of source codepoints, which is what the parser must return.
struct SourceSpan {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;

    bool empty() const noexcept { return end <= begin; }
    bool operator==(const SourceSpan &) const = default;
};

// Maps a span over expanded positions back onto whole source characters.
//
// A model working on expanded positions can put a boundary in the middle of one character —
// halfway through 進's "shin" — and a span that covers half a character is not expressible against
// the original name. Snapping OUTWARD is the safe direction: a partially covered character is
// included whole, so the span always lands on real boundaries and never loses text it selected.
//
// `sourcePositions` must be non-decreasing, which expansion guarantees.
inline SourceSpan snapToSourceCharacters(std::span<const std::uint32_t> sourcePositions,
                                         std::size_t begin,
                                         std::size_t end) {
    if (begin >= end || sourcePositions.empty()) {
        return {};
    }
    end = std::min(end, sourcePositions.size());
    begin = std::min(begin, end - 1);
    // The first and last covered positions already name their own source characters; taking them
    // whole is the snap.
    return {sourcePositions[begin], sourcePositions[end - 1] + 1};
}

// Why one filename was routed Japanese or Chinese.
//
// route() makes exactly one Chinese/Japanese decision per name and then throws the evidence away;
// nothing downstream of the routes can reconstruct it, because the same route array is produced by
// a confident decision and a coin-flip one. A caller that wants to audit, threshold differently, or
// record the score has to be handed it here.
//
// `deterministic` is what separates the two ways a probability of 1.0 arises: Kana anywhere in the
// name selects Japanese without calling the model, so a deterministic 1.0 is an OBSERVATION, while
// a model 1.0 is a score. Reading them as the same number overstates the model.
//
// `evaluated` is false when the name holds no Han or Kana at all. There is no decision then, and
// the zero probability is absence, not evidence of Chinese.
struct CjDecision {
    bool evaluated = false;
    bool deterministic = false;
    bool japanese = false;
    double japaneseProbability = 0.0;
    double japaneseDecisionThreshold = 0.5;
    std::uint32_t hanCodepoints = 0;
    std::uint32_t kanaCodepoints = 0;
};

struct ContextualTrace {
    std::uint32_t matchStart = 0;
    std::uint16_t matchLength = 0;
    std::uint16_t matchOffset = 0;

    bool matched() const noexcept {
        return matchLength != 0;
    }
};

} // namespace neurelease::model::preprocessing
