#include <neurelease/release_parser.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

#ifdef RP_MODELS_DIR_TOKEN
#define RP_STRINGIFY_VALUE_DETAIL(value) #value
#define RP_STRINGIFY_VALUE(value) RP_STRINGIFY_VALUE_DETAIL(value)
constexpr const char* ModelsDirectory = RP_STRINGIFY_VALUE(RP_MODELS_DIR_TOKEN);
#else
constexpr const char* ModelsDirectory = RP_MODELS_DIR;
#endif

struct ParserOwner {
    rp_parser* value = nullptr;
    ParserOwner() = default;
    ParserOwner(const ParserOwner&) = delete;
    ParserOwner& operator=(const ParserOwner&) = delete;
    ParserOwner(ParserOwner&& other) noexcept : value(other.value) { other.value = nullptr; }
    ~ParserOwner() { rp_parser_free(value); }
};

struct ResultOwner {
    rp_result* value = nullptr;
    ~ResultOwner() { rp_result_free(value); }
};

struct BatchOwner {
    rp_batch* value = nullptr;
    ~BatchOwner() { rp_batch_free(value); }
};

ParserOwner parser() {
    ParserOwner result;
    REQUIRE(rp_parser_new(ModelsDirectory, &result.value) == RP_OK);
    REQUIRE(result.value != nullptr);
    return result;
}

} // namespace

TEST_CASE("ABI identity and invalid arguments are stable") {
    CHECK(rp_abi_version() == RP_ABI_VERSION);
    CHECK(rp_parser_new(ModelsDirectory, nullptr) == RP_ERROR_INVALID_ARGUMENT);

    rp_parser* missing = nullptr;
    CHECK(rp_parser_new("directory-that-does-not-exist", &missing) == RP_ERROR_MODEL_NOT_FOUND);
    CHECK(missing == nullptr);
    CHECK(std::strlen(rp_global_error_message()) > 0);

    CHECK(rp_set_accuracy(static_cast<rp_accuracy>(99)) == RP_ERROR_INVALID_ARGUMENT);
    CHECK(rp_set_accuracy(RP_ACCURACY_EXACT) == RP_OK);
    CHECK(rp_get_accuracy() == RP_ACCURACY_EXACT);
}

TEST_CASE("a real learned parse crosses the C boundary as typed values") {
    auto handle = parser();
    ResultOwner result;
    REQUIRE(rp_parse(handle.value,
                     "Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-TERMiNAL.mkv",
                     &result.value) == RP_OK);
    REQUIRE(result.value != nullptr);

    REQUIRE(rp_str(result.value, RP_FIELD_TITLE) != nullptr);
    CHECK(std::string(rp_str(result.value, RP_FIELD_TITLE)).find("Blade Runner") != std::string::npos);
    CHECK(rp_int(result.value, RP_INT_YEAR) == 2017);
    CHECK(rp_int(result.value, RP_INT_TOKENS) > 0);
    CHECK(rp_flag(result.value, RP_FLAG_VALID) == 1);
    CHECK(rp_flag(result.value, RP_FLAG_DEGRADED) == 0);
    CHECK(rp_content(result.value) == RP_CONTENT_MOVIE);
    // The sixth verdict: a live-action film is not anime, and the model was asked, so the
    // confidence is non-zero. Zero confidence would mean a model without the field.
    CHECK(rp_anime(result.value) == 0);
    CHECK(rp_verdict_confidence(result.value, RP_VERDICT_ANIME) > 0.0F);
    CHECK(rp_screen_size(result.value) == RP_RESOLUTION_2160P);
    CHECK(rp_source(result.value) == RP_SOURCE_BLURAY);
    CHECK(rp_video_codec_value(result.value) == RP_CODEC_HEVC);
    CHECK(rp_medium(result.value) == RP_MEDIUM_VIDEO);
    CHECK((rp_special(result.value) == RP_SPECIAL_MOVIE ||
           rp_special(result.value) == RP_SPECIAL_NONE));
    CHECK(rp_flag(result.value, RP_FLAG_SPECIALS) == 0);
    CHECK(rp_adult(result.value) != RP_ADULT_UNKNOWN);
    CHECK(rp_verdict_confidence(result.value, RP_VERDICT_CONTENT) > 0.0F);

    CHECK(rp_edition_at(result.value, rp_edition_count(result.value)) == RP_EDITION_UNKNOWN);
    REQUIRE(rp_release_group_count(result.value) >= 1);
    CHECK(rp_release_group_at(result.value, 0) != nullptr);
    CHECK(rp_release_group_at(result.value, rp_release_group_count(result.value)) == nullptr);
    CHECK(rp_language_at(result.value, rp_language_count(result.value)) == nullptr);
    CHECK(rp_subtitle_language_at(result.value,
                                  rp_subtitle_language_count(result.value)) == nullptr);

    REQUIRE(rp_origin_count(result.value) > 0);
    rp_origin_view origin{};
    CHECK(rp_origin_at(result.value, 0, &origin) == RP_OK);
    CHECK(origin.field >= RP_ORIGIN_TITLE);
    CHECK(origin.field <= RP_ORIGIN_SPECIAL_KIND);
    CHECK(origin.text != nullptr);
    CHECK(rp_origin_at(result.value, rp_origin_count(result.value), &origin) ==
          RP_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("batch results preserve input order and have explicit ownership") {
    auto handle = parser();
    const char* names[]{
        "Game.of.Thrones.S01E08.1080p.WEB-DL.x264-GRP.mkv",
        "Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-TERMiNAL.mkv",
    };
    BatchOwner batch;
    REQUIRE(rp_parse_batch(handle.value, names, 2, &batch.value) == RP_OK);
    REQUIRE(rp_batch_size(batch.value) == 2);
    const rp_result* first = rp_batch_at(batch.value, 0);
    const rp_result* second = rp_batch_at(batch.value, 1);
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(rp_content(first) == RP_CONTENT_SERIES);
    CHECK(rp_content(second) == RP_CONTENT_MOVIE);
    CHECK(rp_batch_at(batch.value, 2) == nullptr);

    BatchOwner empty;
    CHECK(rp_parse_batch(handle.value, nullptr, 0, &empty.value) == RP_OK);
    CHECK(rp_batch_size(empty.value) == 0);
}

TEST_CASE("a null entry rejects the entire batch without a partial result") {
    auto handle = parser();
    const char* names[]{"Movie.2024.mkv", nullptr};
    rp_batch* batch = nullptr;
    CHECK(rp_parse_batch(handle.value, names, 2, &batch) == RP_ERROR_INVALID_ARGUMENT);
    CHECK(batch == nullptr);
    CHECK(std::strlen(rp_last_error_message(handle.value)) > 0);
}

TEST_CASE("rp_view carries every scalar at once, and says which zeros are stated") {
    ParserOwner parser;
    REQUIRE(rp_parser_new(ModelsDirectory, &parser.value) == RP_OK);

    ResultOwner specials;
    REQUIRE(rp_parse(parser.value, "Show.S00E01.Special.1080p.WEB-DL.x264-GRP.mkv",
                     &specials.value) == RP_OK);
    rp_result_view view{};
    REQUIRE(rp_view(specials.value, &view) == RP_OK);
    CHECK(view.season == 0);
    CHECK((view.stated & 2U) != 0U);   // S00: a stated zero, not an absence
    CHECK((view.stated & 8U) != 0U);   // E01 stated
    CHECK(view.valid == 1);

    // REAL S00 NAMES, NOT ONE SYNTHETIC ONE. This case used to assert the stated zero on a single
    // hand-written name, and passed while the parser dropped the season on 27 of 60 real `S00`
    // names in the corpus — the engine read season and episode only from spans the segmenter
    // marked, and the one name here happened to be marked. A written-out `SxxExx` is decidable
    // from the characters, so every spelling the corpus actually carries is asserted.
    struct Special { const char* name; int season; int episode; };
    const Special specialSeasons[] = {
        {"Clannad.After.Story.DVD.Special.2009.s00e03.1080p.BluRay.Opus.2.0-ZR", 0, 3},
        {"Beyond Paradise S00E02 Christmas Special 2024 1080p AMZN WEB-DL", 0, 2},
        {"Two.Doors.Down.S00E05.2022.Christmas.Special.720p.iP.WEBRip.AAC2.0.H264-GRP", 0, 5},
        {"Parks and Recreation S00E12 A Parks and Recreation Special (1080p)", 0, 12},
        {"30.Rock.S00E00.Unaired.Pilot.NBC.Internal.DVD-RIP-DIVX", 0, 0},
        {"[uba] Love, Chunibyo & Other Delusions! - S00E02 - OVA 2 (BD 1080p)", 0, 2},
    };
    for (const Special& special : specialSeasons) {
        ResultOwner parsed;
        REQUIRE(rp_parse(parser.value, special.name, &parsed.value) == RP_OK);
        rp_result_view got{};
        REQUIRE(rp_view(parsed.value, &got) == RP_OK);
        CHECK_MESSAGE(got.season == special.season, special.name);
        CHECK_MESSAGE(got.episode == special.episode, special.name);
        // Both are written zeros or written numbers; either way the origin exists, so `stated`
        // must say so — that is the bit a caller uses to tell `S00` from "no season".
        CHECK_MESSAGE((got.stated & 2U) != 0U, special.name);
        CHECK_MESSAGE((got.stated & 8U) != 0U, special.name);
    }

    // A name with a season and NO episode keeps the episode absent rather than inventing a zero.
    ResultOwner seasonOnly;
    REQUIRE(rp_parse(parser.value,
                     "Archer.2009.S00.Heart.of.Archness.1080p.BluRay.REMUX.AVC.DTS-HD.MA.5.1-GRP",
                     &seasonOnly.value) == RP_OK);
    rp_result_view onlySeason{};
    REQUIRE(rp_view(seasonOnly.value, &onlySeason) == RP_OK);
    CHECK(onlySeason.season == 0);
    CHECK((onlySeason.stated & 2U) != 0U);
    CHECK((onlySeason.stated & 8U) == 0U);

    // A MARKER THAT NAMES NO NUMBER IS NOT A STATED ZERO. The segmenter marks the bare word
    // `Staffel` and leaves the `2` in the title; the mapper records that span with an EMPTY value
    // and no season behind it, and `stated` was reading the origin alone - so the caller was told
    // season 0, confidently, on a name that says season 2. A written S00 above still qualifies
    // because its origin carries the string "0". The missed `2` is the segmenter's and only
    // training moves it; what this pins is that a miss reads as absent rather than as zero.
    ResultOwner wordMarker;
    REQUIRE(rp_parse(parser.value, "Die Simpsons 2. Staffel Folge 5", &wordMarker.value) == RP_OK);
    rp_result_view unnumbered{};
    REQUIRE(rp_view(wordMarker.value, &unnumbered) == RP_OK);
    CHECK((unnumbered.stated & 2U) == 0U);
    CHECK((unnumbered.stated & 8U) != 0U);   // `Folge 5` does name its number
    CHECK(unnumbered.episode == 5);

    ResultOwner movie;
    REQUIRE(rp_parse(parser.value, "Movie.2024.1080p.WEB-DL.x264-GRP.mkv", &movie.value) == RP_OK);
    REQUIRE(rp_view(movie.value, &view) == RP_OK);
    CHECK((view.stated & 2U) == 0U);   // no season anywhere in the name
    CHECK((view.stated & 8U) == 0U);
    CHECK((view.stated & 1U) != 0U);   // the year IS stated
    CHECK(view.year == 2024);
    CHECK(std::string(view.title) == "Movie");
    CHECK(view.origin_count > 0);

    // The one-call origins fill agrees with the per-index accessor.
    std::vector<rp_origin_view> filled(view.origin_count);
    REQUIRE(rp_origins_fill(movie.value, filled.data(), filled.size()) == filled.size());
    rp_origin_view single{};
    REQUIRE(rp_origin_at(movie.value, 0, &single) == RP_OK);
    CHECK(std::string(filled[0].value) == single.value);
    CHECK(filled[0].begin == single.begin);
}

TEST_CASE("invalid single-parse arguments leave diagnostic detail") {
    auto handle = parser();
    rp_result* result = nullptr;
    CHECK(rp_parse(handle.value, nullptr, &result) == RP_ERROR_INVALID_ARGUMENT);
    CHECK(result == nullptr);
    CHECK(std::strlen(rp_last_error_message(handle.value)) > 0);
}
