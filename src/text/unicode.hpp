// The Unicode operations this library actually performs, and no more.
//
// Deliberately NOT a general Unicode library. The parser asks four questions of a codepoint — is it
// a digit, is it a letter, is it Latin, is it from a script that marks a translated title — and one
// of a string: what is its case-folded form for use as a comparison key. Each is answered by a small
// table or range check generated for exactly the scripts release names arrive in, which keeps the
// binary free of ICU and the answers auditable in one screen of ranges.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace neurelease::text {

// --- UTF-8 iteration -------------------------------------------------------------------------------

// Decodes the codepoint starting at byte `at`, advancing `at` past it. Invalid sequences decode as
// U+FFFD and advance one byte, so a malformed name degrades instead of looping.
char32_t decodeAt(std::string_view text, std::size_t& at) noexcept;

// Appends one codepoint as UTF-8.
void appendUtf8(std::string& out, char32_t codepoint);

// --- classification --------------------------------------------------------------------------------

[[nodiscard]] bool isAsciiDigit(char32_t cp) noexcept;
[[nodiscard]] bool isAsciiLetter(char32_t cp) noexcept;

// A LETTER of the Latin script, accents included (ASCII, Latin-1, Latin Extended-A/B). This is what
// decides that "Silo" survives a bilingual title while "第一季" does not.
[[nodiscard]] bool isLatinLetter(char32_t cp) noexcept;

// Any letter, Unicode-wide enough for this domain: Latin plus every foreign script below.
[[nodiscard]] bool isLetter(char32_t cp) noexcept;

// A character from a NON-LATIN script that release names actually arrive in - Han, Hiragana,
// Katakana, Hangul, Cyrillic, Greek, Arabic, Hebrew, Thai, Devanagari - plus the CJ punctuation and
// fullwidth blocks, which is how "：" inside "Spirit Cage：Incarnation" is recognised as foreign.
// The convention it serves: a local site prepends its translation to the real title, and the Latin
// half is the one worth keeping.
[[nodiscard]] bool isForeignScript(char32_t cp) noexcept;

// --- Unicode general categories, from the UCD ------------------------------------------------------
//
// Generated tables, FROZEN to the Unicode database the training tokeniser classified with:
// regenerating them against a newer Unicode would silently move token boundaries and break the
// model contract. The generator lives beside the training pipeline, not here. These predicates
// define the model's
// TOKEN BOUNDARIES and must match the Unicode data the training tokeniser classified with - a
// codepoint categorised differently here moves a boundary and every span after it.

// Category Nd.
[[nodiscard]] bool isDecimalDigit(char32_t cp) noexcept;

// Categories Lu, Ll, Lt, Lm, Lo, Nl, No - the training tokeniser's letter class.
[[nodiscard]] bool isRunLetterCategory(char32_t cp) noexcept;

// --- case ------------------------------------------------------------------------------------------

// ASCII upper/lower, in place over a whole string. Most vocabulary comparisons are ASCII tags and
// need nothing more.
[[nodiscard]] std::string asciiUpper(std::string_view text);
[[nodiscard]] std::string asciiLower(std::string_view text);

// A case-folded COMPARISON KEY: ASCII and Latin-1 folded, everything else passed through. Used for
// dedup keys and vote keys, never for display. Full Unicode folding is deliberately absent - two
// spellings of one title in a non-Latin script differing only by case is not a case release names
// present, and carrying the table for it would be weight without evidence.
[[nodiscard]] std::string foldKey(std::string_view text);

// --- whitespace ------------------------------------------------------------------------------------

// Leading and trailing whitespace removed, inner runs collapsed to one space.
[[nodiscard]] std::string simplified(std::string_view text);

[[nodiscard]] std::string_view trimmed(std::string_view text) noexcept;

} // namespace neurelease::text
