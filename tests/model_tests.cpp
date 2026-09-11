// The segmenter facade, exercised against the real shipped weights: UTF-8 in, byte-offset spans
// out, typed verdicts filled. These are contract tests, not accuracy tests — accuracy is scored
// against the gold corpus in the training repository.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "model/segmenter.hpp"

#include <string>

using namespace neurelease;
using neurelease::Analysis;
using neurelease::model::NameSegmenter;

namespace {
#ifdef RP_MODELS_DIR_TOKEN
#define RP_STRINGIFY_VALUE_DETAIL(value) #value
#define RP_STRINGIFY_VALUE(value) RP_STRINGIFY_VALUE_DETAIL(value)
constexpr const char* ModelsDirectory = RP_STRINGIFY_VALUE(RP_MODELS_DIR_TOKEN);
#else
constexpr const char* ModelsDirectory = RP_MODELS_DIR;
#endif

NameSegmenter& segmenter() {
    static NameSegmenter instance(ModelsDirectory);
    return instance;
}
} // namespace

TEST_CASE("a plain episode name segments with byte offsets that index the source") {
    const std::string name = "Game.of.Thrones.S01E08.1080p.WEB-DL.x264-GRP";
    const Analysis analysis = segmenter().analyze(name);
    REQUIRE(analysis.valid);
    REQUIRE(!analysis.spans.empty());
    for (const auto& span : analysis.spans) {
        CHECK(span.begin >= 0);
        CHECK(span.end > span.begin);
        CHECK(span.end <= static_cast<int>(name.size()));
    }
    // The title is the first thing the model reads, and its bytes are the bytes of the name.
    const auto& first = analysis.spans.front();
    CHECK(first.type == SpanType::Main);
    CHECK(first.typeLabel == spanTypeName(first.type));
    CHECK(name.substr(first.begin, first.end - first.begin).starts_with("Game"));
    CHECK(analysis.tokens > 0);
    CHECK(analysis.content == ContentKind::Series);
    CHECK(analysis.numbering == NumberingKind::SeasonEpisode);
}

TEST_CASE("multi-byte characters keep spans on codepoint boundaries") {
    // 灵笼 is 3 bytes per character; every span edge must land between codepoints, never inside.
    const std::string name =
        "\xE7\x81\xB5\xE7\xAC\xBC Ling Cage 2019 S01 1080p WEB-DL H264 AAC-HQC";
    const Analysis analysis = segmenter().analyze(name);
    REQUIRE(analysis.valid);
    for (const auto& span : analysis.spans) {
        // A continuation byte (10xxxxxx) at a span edge means the edge split a codepoint.
        if (span.begin < static_cast<int>(name.size()))
            CHECK((static_cast<unsigned char>(name[span.begin]) & 0xC0) != 0x80);
        if (span.end < static_cast<int>(name.size()))
            CHECK((static_cast<unsigned char>(name[span.end]) & 0xC0) != 0x80);
    }
    CHECK(analysis.languageEvaluated);
}

TEST_CASE("a movie is the movie entry, not a special") {
    const Analysis analysis =
        segmenter().analyze("Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-TERMiNAL");
    REQUIRE(analysis.valid);
    CHECK(analysis.content == ContentKind::Movie);
    // The head answers "which entry is this" — on a feature film that answer is Movie, and it
    // must never be read as "this is a special".
    CHECK((analysis.special == SpecialKind::Movie || analysis.special == SpecialKind::None));
}

TEST_CASE("a malformed byte degrades to one replacement codepoint, offsets still index the source") {
    // The decoder turns each bad byte into one U+FFFD and carries on; spans must keep addressing
    // the ORIGINAL bytes, not a re-encoded copy where the replacement would be three bytes wide.
    const std::string bad = "Movie.\xC3.2024.1080p.WEB-DL-GRP"; // truncated two-byte sequence
    const Analysis analysis = segmenter().analyze(bad);
    REQUIRE(analysis.valid);
    for (const auto& span : analysis.spans) {
        CHECK(span.begin >= 0);
        CHECK(span.end <= static_cast<int>(bad.size()));
        const std::string text = bad.substr(span.begin, span.end - span.begin);
        if (span.type == SpanType::Year) CHECK(text == "2024");
        if (span.type == SpanType::Resolution) CHECK(text == "1080p");
    }
}

TEST_CASE("an overlong name is invalid, not a crash") {
    const std::string overlong(20000, 'a'); // past the 16 KB preprocessing bound
    const Analysis analysis = segmenter().analyze(overlong);
    CHECK_FALSE(analysis.valid);
    CHECK(analysis.spans.empty());
    CHECK(analysis.content == ContentKind::Unknown);
}

TEST_CASE("verdict labels and typed verdicts agree") {
    const Analysis analysis =
        segmenter().analyze("The.Wire.S01-S05.COMPLETE.1080p.BluRay.x265-Grp");
    REQUIRE(analysis.valid);
    CHECK(analysis.pack != PackScope::Unknown);
    CHECK(analysis.adult != AdultKind::Unknown);
    CHECK(analysis.contentKind.confidence > 0.0F);
    CHECK(!analysis.kernelPath.empty());
}
