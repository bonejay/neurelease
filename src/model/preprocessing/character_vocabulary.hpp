// The character encoder's vocabulary layout: which embedding row each kind of token occupies.
// These are architecture constants of the trained model (pretraining/fields.py in the training
// repository defines them, encoder.cpp repeats the row count); the READING TABLES that map a
// codepoint to its letters are model data and load from model/character_map.bin instead.
#pragma once

#include <cstdint>

namespace neurelease::model::preprocessing::vocabulary {

inline constexpr std::uint16_t kVocabularyRows = 150;
inline constexpr std::uint16_t kPadding = 0;
inline constexpr std::uint16_t kMask = 1;
inline constexpr std::uint16_t kAsciiFirst = 2;
inline constexpr std::uint16_t kTransliteratedFirst = 97;
inline constexpr std::uint16_t kOriginHanGeneric = 123;
inline constexpr std::uint16_t kOriginJapanese = 124;
inline constexpr std::uint16_t kOriginKorean = 125;
inline constexpr std::uint16_t kOriginCyrillic = 126;
inline constexpr std::uint16_t kLatinModified = 127;
inline constexpr std::uint16_t kLatinLigature = 128;
inline constexpr std::uint16_t kSourceUppercase = 136;
inline constexpr std::uint16_t kUnknownCharacter = 137;
inline constexpr std::uint16_t kBeginName = 138;
inline constexpr std::uint16_t kEndName = 139;

} // namespace neurelease::model::preprocessing::vocabulary
