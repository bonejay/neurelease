// The text layer is the ground everything else stands on: every vocabulary table and every reader
// types against these semantics, so they are pinned here first.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "text/regex.hpp"
#include "text/unicode.hpp"

using namespace neurelease::text;

TEST_CASE("a literal pattern matches and reports byte offsets") {
    static const Regex re(R"(S(\d{1,2})(?:[ ._-]*E(\d{1,4}))?)", true);
    const std::string name = "Game.of.Thrones.S01 E08.1080p";
    const Match match = re.match(name);
    REQUIRE(match.hasMatch());
    CHECK(match.captured(1) == "01");
    CHECK(match.captured(2) == "08");
    CHECK(match.capturedStart(0) == 16);
}

TEST_CASE("lookbehind and lookahead survive the port") {
    // The resolution reader's shape: a number that is not part of a longer number.
    static const Regex re(R"((?<!\d)(\d{3,4})[pP](?!\w))");
    CHECK(re.match("Movie.1080p.BluRay").captured(1) == "1080");
    CHECK_FALSE(re.match("Movie.21080p.BluRay").hasMatch());
}

TEST_CASE("UTF-8 subjects and codepoint escapes address codepoints") {
    static const Regex re(R"([\x{FF01}-\x{FF60}])");
    const std::string name = "Spirit Cage\xEF\xBC\x9AIncarnation"; // U+FF1A fullwidth colon
    const Match match = re.match(name);
    REQUIRE(match.hasMatch());
    CHECK(match.capturedStart(0) == 11);
    CHECK(match.captured(0) == "\xEF\xBC\x9A");
}

TEST_CASE("case insensitivity is an option, not a default") {
    static const Regex sensitive(R"(WEB-DL)");
    static const Regex insensitive(R"(WEB-DL)", true);
    CHECK_FALSE(sensitive.match("movie.web-dl.x264").hasMatch());
    CHECK(insensitive.match("movie.web-dl.x264").hasMatch());
}

TEST_CASE("the global-match loop advances by capturedEnd") {
    static const Regex re(R"(\d{4})");
    const std::string text = "1999 2005 2019";
    int found = 0;
    std::size_t from = 0;
    while (true) {
        const Match match = re.match(text, from);
        if (!match.hasMatch()) break;
        ++found;
        from = match.capturedEnd() + (match.capturedStart() == match.capturedEnd() ? 1 : 0);
    }
    CHECK(found == 3);
}

TEST_CASE("an invalid pattern throws instead of never matching") {
    CHECK_THROWS_AS(Regex(R"((unclosed)"), std::invalid_argument);
}

TEST_CASE("utf8 decoding walks real names") {
    const std::string name = "\xE7\x81\xB5\xE7\xAC\xBC Ling Cage"; // 灵笼 Ling Cage
    std::size_t at = 0;
    CHECK(decodeAt(name, at) == char32_t{0x7075});
    CHECK(decodeAt(name, at) == char32_t{0x7B3C});
    CHECK(decodeAt(name, at) == char32_t{' '});
    CHECK(at == 7);
}

TEST_CASE("malformed utf8 degrades to one replacement character per bad byte") {
    const std::string bad = "a\xC3"; // truncated two-byte sequence at the end
    std::size_t at = 0;
    CHECK(decodeAt(bad, at) == char32_t{'a'});
    CHECK(decodeAt(bad, at) == char32_t{0xFFFD});
    CHECK(at == bad.size());
}

TEST_CASE("script classification matches the title rule's needs") {
    CHECK(isForeignScript(0x7075));  // 灵 Han
    CHECK(isForeignScript(0x0417));  // З Cyrillic
    CHECK(isForeignScript(0xC624));  // 오 Hangul
    CHECK(isForeignScript(0xFF1A));  // ： fullwidth colon
    CHECK_FALSE(isForeignScript('S'));
    CHECK_FALSE(isForeignScript(0xE9)); // é stays Latin
    CHECK(isLatinLetter(0xE9));
    CHECK_FALSE(isLatinLetter(0xD7)); // × is not a letter
}

TEST_CASE("simplified collapses runs and foldKey folds Latin case") {
    CHECK(simplified("  a\t\tb  c  ") == "a b c");
    CHECK(foldKey("BluRay") == "bluray");
    CHECK(foldKey("\xC3\x89t\xC3\xA9") == "\xC3\xA9t\xC3\xA9"); // Été -> été
    CHECK(trimmed("  x  ") == "x");
}
