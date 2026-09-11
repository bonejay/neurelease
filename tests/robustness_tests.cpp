// The public Parser, fed what a real index feeds it: names in five scripts, and the inputs
// nobody plans for — empty, whitespace, separators only, NUL bytes, emoji, a five-thousand
// character name. The language cases pin end-to-end behaviour the unit suites only cover
// implicitly through the reference activations; the hostile cases pin one contract: garbage in,
// `valid == false` out, never a crash and never a fabricated parse.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "neurelease/parser.hpp"

#include <string>

using namespace neurelease;

namespace {
#ifdef RP_MODELS_DIR_TOKEN
#define RP_STRINGIFY_VALUE_DETAIL(value) #value
#define RP_STRINGIFY_VALUE(value) RP_STRINGIFY_VALUE_DETAIL(value)
constexpr const char* ModelsDirectory = RP_STRINGIFY_VALUE(RP_MODELS_DIR_TOKEN);
#else
constexpr const char* ModelsDirectory = RP_MODELS_DIR;
#endif

Parser& parser() {
    static Parser instance{ModelsDirectory};
    return instance;
}
} // namespace

TEST_CASE("a German dual-language release reads as German, not as a source") {
    const ReleaseInfo info =
        parser().parse("Die.Discounter.S04E03.German.DL.2160p.WEB.h265-ohSNAP").info;
    CHECK(info.title == "Die Discounter"); // the article stays in the title
    CHECK(info.season == 4);
    CHECK(info.episode == 3);
    REQUIRE(info.languages.size() == 1);
    CHECK(info.languages.front() == "deu");
    CHECK(info.dualAudio); // DL is the dual-language flag, not a source
    CHECK(info.releaseGroup == "ohSNAP");
}

TEST_CASE("a Cyrillic title survives with its numbering and language") {
    const ReleaseInfo info = parser().parse("Эпидемия.S02E05.2022.WEB-DL.1080p.RUS.LostFilm").info;
    CHECK(info.title == "Эпидемия");
    CHECK(info.season == 2);
    CHECK(info.episode == 5);
    CHECK(info.year == 2022);
    REQUIRE(!info.languages.empty());
    CHECK(info.languages.front() == "rus");
    CHECK(info.releaseGroup == "LostFilm");
}

TEST_CASE("a Japanese fansub name keeps the native title and absolute numbering") {
    const ReleaseInfo info =
        parser().parse("[ANi] 葬送のフリーレン - 28 [1080P][Baha][WEB-DL][AAC AVC][CHT][MP4]").info;
    CHECK(info.title == "葬送のフリーレン");
    CHECK(info.absoluteEpisode == 28);
    CHECK(info.numbering == NumberingKind::Absolute);
    CHECK(info.releaseGroup == "ANi");
    CHECK(info.streamingService == "BAHA");
}

TEST_CASE("a Han-script group is a group, not a title") {
    const ReleaseInfo info =
        parser().parse("[桜都字幕组] 葬送的芙莉莲 / Sousou no Frieren [28][1080p][简体内嵌]").info;
    CHECK(info.title == "Sousou no Frieren"); // romanization is the main title
    CHECK(info.releaseGroup == "桜都字幕组");
    CHECK(info.absoluteEpisode == 28);
}

TEST_CASE("a French language tag is a language, not a title word") {
    const ReleaseInfo info =
        parser().parse("Les.Miserables.2019.FRENCH.1080p.BluRay.x264-VENUE").info;
    CHECK(info.title == "Les Miserables");
    REQUIRE(!info.languages.empty());
    CHECK(info.languages.front() == "fra");
}

TEST_CASE("hostile inputs are invalid, never a crash and never a parse") {
    // The NUL case is built explicitly: a "\0abc" literal truncates at the NUL and would silently
    // test the empty string twice.
    for (const std::string name : {std::string{}, std::string{" "}, std::string{"\t\n"},
                                   std::string{"..."}, std::string{"___"}, std::string{"()[]{}-"},
                                   std::string{"\0abc", 4},
                                   std::string{"\xF0\x9F\x8E\xAC\xF0\x9F\x8E\xAC"}}) {
        const ReleaseInfo info = parser().parse(name).info;
        CAPTURE(name);
        CHECK(!info.valid);
        CHECK(info.title.empty());
    }
}

TEST_CASE("malformed UTF-8 degrades instead of crashing") {
    // A lone continuation byte and a truncated sequence inside an otherwise ordinary name.
    const ReleaseInfo info = parser().parse("Movie\x80.2024.1080p.WEB-DL.x264-GRP\xC3").info;
    CHECK(info.year == 2024);
    CHECK(info.screenSize == ResolutionTier::P1080);
}

TEST_CASE("technical facts survive far beyond the position table") {
    // The token Transformer's position tables hold 96 slots per direction; past them the
    // from-start index clamps. This name has ~119 runs (over 230 tokens), so most of it sits
    // beyond the table -- and the technical block still reads, wherever it stands: the character
    // CNN is position-free, and distance-from-the-end stays exact for the last 96 tokens.
    // WHAT DOES NOT SURVIVE, measured rather than assumed: the title. A hundred-word filler is a
    // shape training never saw, and the title head loses it -- so nothing here asserts one.
    std::string filler;
    for (int index = 0; index < 110; ++index) {
        filler += "word" + std::to_string(index) + ".";
    }
    for (const std::string name : {
             "My.Show." + filler + "S01E03.1080p.WEB-DL.x264-GRP.mkv",   // tech block at the end
             "My.Show.S01E03.1080p." + filler + "WEB-DL.x264-GRP.mkv",   // tech block split around it
         }) {
        CAPTURE(name.substr(0, 60));
        const ReleaseInfo info = parser().parse(name).info;
        CHECK(info.season == 1);
        CHECK(info.episode == 3);
        CHECK(info.screenSize == ResolutionTier::P1080);
        CHECK(info.releaseGroup == "GRP");
        CHECK(info.container == "mkv");
    }
}

TEST_CASE("what is SUPPOSED to fail, fails the documented way") {
    // Negative contracts, pinned as hard as the positive ones: when these start "passing", either
    // the model genuinely improved (retrain, then move the expectation into the positive test
    // above) or something upstream broke the failure path - both are worth a red test.

    SUBCASE("over the 16 KB byte cap the model is skipped, visibly") {
        const std::string base = "Movie.2024.1080p.";
        const std::string over = base + std::string(16 * 1024 - base.size() + 1, 'a');
        const ReleaseInfo info = parser().parse(over).info;
        CHECK(info.degraded);              // the fallback ran, and says so
        CHECK(info.year == 2024);          // the title/year heuristics still answer
        CHECK(info.screenSize == ResolutionTier::Unknown); // the model never saw the name
    }

    SUBCASE("a hundred-word filler defeats the title head") {
        std::string filler;
        for (int index = 0; index < 110; ++index) {
            filler += "word" + std::to_string(index) + ".";
        }
        const ReleaseInfo info =
            parser().parse("My.Show." + filler + "S01E03.1080p.WEB-DL.x264-GRP.mkv").info;
        // THE TITLE HEAD IS DEFEATED EITHER WAY. Earlier weights returned no title and an invalid
        // parse; the 2026-09-07 weights read the markers at the end and call a filler word the
        // title. Both are the same failure - the title is out of reach - so the test pins that and
        // not which form it takes, and it pins that the markers beyond the filler still survive.
        CHECK(info.title != "My Show");
        CHECK(!info.degraded);      // the model DID run - this is a miss, not the fallback
        if (info.valid) {
            CHECK(info.season == 1);
            CHECK(info.episode == 3);
        }
    }

    SUBCASE("a name that is one technical block carries no title to find") {
        const ReleaseInfo info = parser().parse("1080p.WEB-DL.x264.AAC.5.1.HEVC.REMUX").info;
        CHECK(info.title.empty());
    }
}

TEST_CASE("a five-thousand character name parses or refuses, in bounded time") {
    const std::string absurd = std::string(5000, 'a') + ".2024.1080p.mkv";
    const ReleaseInfo info = parser().parse(absurd).info; // must return, whatever it decides
    CHECK((info.valid || info.title.empty()));
}

TEST_CASE("the batch answers exactly like the single-name path") {
    const std::string name = "Show.S01E03.1080p.WEB-DL.x264-GRP.mkv";
    BatchParser batch{ModelsDirectory, 2};
    const std::vector<std::string> names{name, name, name};
    const auto results = batch.parse(names);
    REQUIRE(results.size() == 3);
    const ReleaseInfo single = parser().parse(name).info;
    for (const auto& result : results) {
        CHECK(result.info == single);
    }
}
