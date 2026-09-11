// Per-codepoint reading tables: one Han, Kana, Hangul, Cyrillic or modified-Latin character to the
// Latin letters the model was trained to see for it. Model data, not source: they were generated
// beside the weights and must match them, so they ship as model/character_map.bin under the
// manifest and load at construction like the other model files.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace neurelease::model::preprocessing {

struct ReadingTable {
    std::vector<std::uint32_t> codepoints; // sorted
    std::vector<std::uint64_t> readings;   // packed letters, parallel to codepoints

    [[nodiscard]] std::optional<std::uint64_t> find(std::uint32_t codepoint) const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
};

struct CharacterTables {
    ReadingTable hanGeneric;
    ReadingTable hanJapanese;
    ReadingTable kanaJapanese;
    ReadingTable hangul;
    ReadingTable cyrillic;
    ReadingTable latinModified;

    // RCTBL1: magic, u32 version 1, u32 table count; per table u8 name length, name, u32 rows,
    // rows x (u32 codepoint, u64 packed reading). Every table above must be present.
    static CharacterTables load(const std::filesystem::path &path);
    [[nodiscard]] std::size_t bytes() const noexcept;
};

} // namespace neurelease::model::preprocessing
