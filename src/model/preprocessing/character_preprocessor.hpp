#pragma once

#include "model/preprocessing/character_tables.hpp"
#include "model/preprocessing/cj_detector.hpp"
#include "model/preprocessing/transliterator.hpp"
#include "model/preprocessing/types.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace neurelease::model::preprocessing {

struct PreparedName {
    std::vector<std::uint32_t> codepoints;
    std::vector<Route> routes;
    CjDecision cj;
};

struct PreprocessedName {
    std::vector<TokenBag> tokens;
    bool valid = false;

    explicit operator bool() const noexcept { return valid; }
};

// Owning return-by-value batch result with one flat token allocation. Row offsets avoid the
// per-name heap allocation of vector<vector<TokenBag>> while preserving input order.
struct PreprocessedBatch {
    std::vector<TokenBag> tokens;
    std::vector<std::size_t> offsets;
    std::vector<std::uint8_t> valid;

    std::size_t size() const noexcept { return valid.size(); }
    bool empty() const noexcept { return valid.empty(); }
    bool succeeded(std::size_t index) const noexcept { return valid[index] != 0; }
    std::span<const TokenBag> row(std::size_t index) const noexcept {
        return std::span(tokens).subspan(offsets[index], offsets[index + 1] - offsets[index]);
    }
};

// Converts one UTF-8 release name into fixed-position model input.
//
// The pipeline decodes UTF-8, routes codepoints by Unicode script, concatenates all Han/Kana in one
// filename for a single Chinese/Japanese decision, and emits one bag of embedding IDs per source
// codepoint. Transliteration still operates on contiguous source spans. BOS/EOS are model-only
// positions around that source-aligned sequence. Tables and scratch storage are reused.
class CharacterPreprocessor {
    public:
    explicit CharacterPreprocessor(const std::filesystem::path &characterMap,
                                   const std::filesystem::path &chineseJapaneseModel,
                                   const std::filesystem::path &japaneseContextualModel = {},
                                   const std::filesystem::path &chineseContextualModel = {});

    bool decode(std::string_view utf8,
                std::vector<std::uint32_t> &codepoints,
                Counters *counters = nullptr) const;

    // `decision` receives the filename-global Chinese/Japanese evidence. It is optional because
    // ordinary preprocessing only needs the routes; passing it also counts Han and Kana, which the
    // routing itself has no use for.
    void route(std::span<const std::uint32_t> codepoints,
               std::vector<Route> &routes,
               Counters *counters = nullptr,
               CjDecision *decision = nullptr);

    void emit(std::span<const std::uint32_t> codepoints,
              std::span<const Route> routes,
              std::vector<TokenBag> &output,
              Counters *counters = nullptr,
              std::vector<ContextualTrace> *contextualTrace = nullptr) const;

    PreprocessedName process(std::string_view utf8);
    PreprocessedName process(std::string_view utf8, Counters &counters);

    PreprocessedBatch processBatch(std::span<const std::string> names);

    PreprocessedBatch processBatch(std::span<const std::string> names, Counters &counters);

    // Lays every letter of a bag on its own position, keeping reading order.
    //
    // A separate step rather than a mode inside emit(), so the compact output stays exactly what
    // it was and the two representations can be compared instead of one replacing the other.
    // 進 is one position holding {s,h,i,n,japanese}; expanded it is four positions, each carrying
    // its letter and the Japanese origin, with the first also carrying CharacterStartRow. That
    // makes a Japanese title the same alphabet as an authored Latin one, which is the whole point.
    //
    // Positions with no letters — boundaries, sentinels, unmapped input — pass through unchanged.
    ExpandedName expand(std::span<const TokenBag> compact) const;

    PreparedName prepare(std::string_view utf8);
    bool containsNonLatinScript(std::string_view utf8);

    ChineseJapaneseDetector &chineseJapaneseDetector() noexcept;
    const ChineseJapaneseDetector &chineseJapaneseDetector() const noexcept;

    std::size_t reusableScratchBytes() const noexcept;
    std::uintmax_t contextualModelFileBytes() const noexcept;
    std::size_t contextualRuntimeTableBytes() const noexcept;
    std::size_t characterTableBytes() const noexcept;

    private:
    // Appends instead of replacing, which is what lets a batch emit straight into its own flat
    // token buffer. emit() is the clearing wrapper around it.
    void appendTokens(std::span<const std::uint32_t> codepoints,
                      std::span<const Route> routes,
                      std::vector<TokenBag> &output,
                      Counters *counters,
                      std::vector<ContextualTrace> *contextualTrace) const;
    bool processOne(std::string_view utf8,
                    std::vector<TokenBag> &output,
                    Counters *counters);
    PreprocessedBatch processBatchImpl(std::span<const std::string> names, Counters *counters);

    CharacterTables tables_;

    ChineseJapaneseDetector chineseJapaneseDetector_;
    std::optional<ContextualTransliterator> japaneseContextual_;
    std::optional<ContextualTransliterator> chineseContextual_;
    std::vector<std::uint32_t> codepoints_;
    std::vector<std::uint32_t> cjScratch_;
    std::vector<std::uint32_t> cjPositions_;
    std::vector<Route> routes_;
    mutable std::vector<ContextualPosition> contextualSpanScratch_;
};


} // namespace neurelease::model::preprocessing
