#include "model/preprocessing/character_tables.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace neurelease::model::preprocessing {
namespace {

template <typename T> T read(std::ifstream &input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated character map");
    return value;
}

} // namespace

std::optional<std::uint64_t> ReadingTable::find(std::uint32_t codepoint) const noexcept {
    const auto found = std::lower_bound(codepoints.begin(), codepoints.end(), codepoint);
    if (found == codepoints.end() || *found != codepoint) return std::nullopt;
    return readings[static_cast<std::size_t>(found - codepoints.begin())];
}

std::size_t ReadingTable::bytes() const noexcept {
    return codepoints.size() * sizeof(std::uint32_t) + readings.size() * sizeof(std::uint64_t);
}

CharacterTables CharacterTables::load(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open character map " + path.string());
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (std::memcmp(magic, "RCTBL1\0\0", sizeof(magic)) != 0)
        throw std::runtime_error("invalid character map magic in " + path.string());
    if (read<std::uint32_t>(input) != 1)
        throw std::runtime_error("unsupported character map version in " + path.string());
    const auto count = read<std::uint32_t>(input);

    std::map<std::string, ReadingTable> tables;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto nameLength = read<std::uint8_t>(input);
        std::string name(nameLength, '\0');
        input.read(name.data(), nameLength);
        const auto rows = read<std::uint32_t>(input);
        ReadingTable table;
        table.codepoints.reserve(rows);
        table.readings.reserve(rows);
        for (std::uint32_t row = 0; row < rows; ++row) {
            table.codepoints.push_back(read<std::uint32_t>(input));
            table.readings.push_back(read<std::uint64_t>(input));
        }
        if (!std::is_sorted(table.codepoints.begin(), table.codepoints.end()))
            throw std::runtime_error("character map table " + name + " is not sorted");
        tables.emplace(name, std::move(table));
    }

    CharacterTables out;
    const auto take = [&tables, &path](const char *name) {
        const auto found = tables.find(name);
        if (found == tables.end())
            throw std::runtime_error("character map " + path.string() + " lacks table " + name);
        return std::move(found->second);
    };
    out.hanGeneric = take("han_generic");
    out.hanJapanese = take("han_japanese");
    out.kanaJapanese = take("kana_japanese");
    out.hangul = take("hangul");
    out.cyrillic = take("cyrillic");
    out.latinModified = take("latin_modified");
    return out;
}

std::size_t CharacterTables::bytes() const noexcept {
    return hanGeneric.bytes() + hanJapanese.bytes() + kanaJapanese.bytes() + hangul.bytes() +
           cyrillic.bytes() + latinModified.bytes();
}

} // namespace neurelease::model::preprocessing
