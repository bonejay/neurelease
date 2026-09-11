#include "text/unicode.hpp"

namespace neurelease::text {

namespace {
#include "unicode_tables.inc"

bool inSortedRanges(char32_t cp, const CodepointRange* ranges, std::size_t count) noexcept {
    // Binary search: the letter table is ~750 ranges, far past the linear-scan comfort zone the
    // hand-written script table lives in.
    std::size_t low = 0, high = count;
    while (low < high) {
        const std::size_t mid = (low + high) / 2;
        if (cp < ranges[mid].begin) high = mid;
        else if (cp > ranges[mid].end) low = mid + 1;
        else return true;
    }
    return false;
}
} // namespace

bool isDecimalDigit(char32_t cp) noexcept {
    if (cp < 0x80) return cp >= '0' && cp <= '9'; // the fast path is the common path
    return inSortedRanges(cp, kDecimalDigitRanges,
                          sizeof(kDecimalDigitRanges) / sizeof(kDecimalDigitRanges[0]));
}

bool isRunLetterCategory(char32_t cp) noexcept {
    if (cp < 0x80) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    return inSortedRanges(cp, kRunLetterRanges,
                          sizeof(kRunLetterRanges) / sizeof(kRunLetterRanges[0]));
}

char32_t decodeAt(std::string_view text, std::size_t& at) noexcept {
    const auto byte = [&](std::size_t offset) {
        return static_cast<unsigned char>(text[at + offset]);
    };
    const unsigned char lead = byte(0);
    // The continuation checks are real, not assumed: a truncated or malformed sequence yields
    // U+FFFD and advances ONE byte, so bad input costs one replacement character, never a loop.
    if (lead < 0x80) {
        ++at;
        return lead;
    }
    if ((lead & 0xE0) == 0xC0 && at + 1 < text.size() && (byte(1) & 0xC0) == 0x80) {
        const char32_t cp = ((lead & 0x1Fu) << 6) | (byte(1) & 0x3Fu);
        at += 2;
        return cp;
    }
    if ((lead & 0xF0) == 0xE0 && at + 2 < text.size() && (byte(1) & 0xC0) == 0x80 &&
        (byte(2) & 0xC0) == 0x80) {
        const char32_t cp = ((lead & 0x0Fu) << 12) | ((byte(1) & 0x3Fu) << 6) | (byte(2) & 0x3Fu);
        at += 3;
        return cp;
    }
    if ((lead & 0xF8) == 0xF0 && at + 3 < text.size() && (byte(1) & 0xC0) == 0x80 &&
        (byte(2) & 0xC0) == 0x80 && (byte(3) & 0xC0) == 0x80) {
        const char32_t cp = ((lead & 0x07u) << 18) | ((byte(1) & 0x3Fu) << 12) |
                            ((byte(2) & 0x3Fu) << 6) | (byte(3) & 0x3Fu);
        at += 4;
        return cp;
    }
    ++at;
    return 0xFFFD;
}

void appendUtf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool isAsciiDigit(char32_t cp) noexcept { return cp >= '0' && cp <= '9'; }

bool isAsciiLetter(char32_t cp) noexcept {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

bool isLatinLetter(char32_t cp) noexcept {
    if (isAsciiLetter(cp)) return true;
    // Latin-1 letters, the two multiplication/division signs excluded.
    if (cp >= 0xC0 && cp <= 0xFF) return cp != 0xD7 && cp != 0xF7;
    // Latin Extended-A and -B: the accented forms European titles actually use.
    return cp >= 0x100 && cp <= 0x24F;
}

namespace {

struct Range {
    char32_t begin;
    char32_t end; // inclusive
};

// The scripts release names arrive in, as codepoint ranges. Hand-checked against the Unicode block
// charts rather than generated from the UCD: ten scripts, a screenful of ranges, auditable at a
// glance - which a 1,500-row generated table would not be. Unassigned codepoints inside a block
// classify with the block, which is exactly the safe direction for "is this foreign?".
constexpr Range kForeign[] = {
    {0x0370, 0x03FF}, // Greek
    {0x0400, 0x04FF}, // Cyrillic
    {0x0500, 0x052F}, // Cyrillic Supplement
    {0x0590, 0x05FF}, // Hebrew
    {0x0600, 0x06FF}, // Arabic
    {0x0750, 0x077F}, // Arabic Supplement
    {0x0900, 0x097F}, // Devanagari
    {0x0E00, 0x0E7F}, // Thai
    {0x1100, 0x11FF}, // Hangul Jamo
    {0x3000, 0x303F}, // CJK symbols and punctuation (、。「」 report as Common in the UCD)
    {0x3040, 0x309F}, // Hiragana
    {0x30A0, 0x30FF}, // Katakana
    {0x3130, 0x318F}, // Hangul compatibility Jamo
    {0x31F0, 0x31FF}, // Katakana phonetic extensions
    {0x3400, 0x4DBF}, // CJK extension A
    {0x4E00, 0x9FFF}, // CJK unified ideographs
    {0xA960, 0xA97F}, // Hangul Jamo extended-A
    {0xAC00, 0xD7FF}, // Hangul syllables (+ Jamo extended-B)
    {0xF900, 0xFAFF}, // CJK compatibility ideographs
    {0xFF01, 0xFF60}, // fullwidth forms - the "：" in "Spirit Cage：Incarnation"
    {0xFF61, 0xFF9F}, // halfwidth Katakana
    {0x20000, 0x2A6DF}, // CJK extension B
};

bool inRanges(char32_t cp, const Range* ranges, std::size_t count) noexcept {
    // Linear over ~22 ranges: at most a handful of comparisons for the common early-out (Latin
    // text is below every range), and no binary-search subtlety to get wrong.
    for (std::size_t at = 0; at < count; ++at) {
        if (cp < ranges[at].begin) return false;
        if (cp <= ranges[at].end) return true;
    }
    return false;
}

} // namespace

bool isForeignScript(char32_t cp) noexcept {
    return inRanges(cp, kForeign, sizeof(kForeign) / sizeof(kForeign[0]));
}

bool isLetter(char32_t cp) noexcept {
    if (isLatinLetter(cp)) return true;
    if (cp == 0x3000 || (cp >= 0xFF01 && cp <= 0xFF20)) return false; // punctuation blocks
    return isForeignScript(cp);
}

std::string asciiUpper(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    return out;
}

std::string asciiLower(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}

std::string foldKey(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t at = 0;
    while (at < text.size()) {
        char32_t cp = decodeAt(text, at);
        if (cp >= 'A' && cp <= 'Z') cp = cp - 'A' + 'a';
        // Latin-1 capitals fold to their lowercase forms; 0xDF (ß) and 0xD7 (×) have none.
        else if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) cp += 0x20;
        appendUtf8(out, cp);
    }
    return out;
}

std::string simplified(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool pendingSpace = false;
    bool started = false;
    for (const char c : text) {
        const bool space = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        if (space) {
            pendingSpace = started;
            continue;
        }
        if (pendingSpace) {
            out.push_back(' ');
            pendingSpace = false;
        }
        out.push_back(c);
        started = true;
    }
    return out;
}

std::string_view trimmed(std::string_view text) noexcept {
    const auto isSpace = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (!text.empty() && isSpace(text.front())) text.remove_prefix(1);
    while (!text.empty() && isSpace(text.back())) text.remove_suffix(1);
    return text;
}

} // namespace neurelease::text
