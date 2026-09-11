#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "model/mapping.hpp"
#include "model/schema.hpp"

#include <algorithm>
#include <string>

using namespace neurelease;
using namespace neurelease::model;

namespace {

SegmentedSpan span(std::string_view name, SpanType type, std::string_view token,
                   float confidence = 0.9F) {
    const std::size_t begin = name.find(token);
    REQUIRE(begin != std::string_view::npos);
    return {type, {}, static_cast<std::int32_t>(begin),
            static_cast<std::int32_t>(begin + token.size()), confidence};
}

Analysis analysisOf(std::string_view name) {
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {
        span(name, SpanType::Main, "Movie"), span(name, SpanType::Year, "2024"),
        span(name, SpanType::Resolution, "2160p"), span(name, SpanType::SourceType, "UHD"),
        span(name, SpanType::Platform, "AMZN"), span(name, SpanType::SourceType, "WEB-DL"),
        span(name, SpanType::Codec, "HEVC"), span(name, SpanType::BitDepth, "10bit"),
        span(name, SpanType::Hdr, "DV"), span(name, SpanType::AudioCodec, "DDP"),
        span(name, SpanType::AudioChannels, "5.1"), span(name, SpanType::AudioFeature, "Atmos"),
        span(name, SpanType::ReleaseGroup, "GROUP"), span(name, SpanType::FileExtension, "mkv"),
    };
    analysis.contentKind = {"movie", 0.95F};
    analysis.animeKind = {"False", 0.93F};
    analysis.numberingKind = {"none", 0.8F};
    analysis.packScope = {"single", 0.9F};
    analysis.specialKind = {"movie", 0.55F};
    return analysis;
}

} // namespace

TEST_CASE("a model 2 content kind still resolves") {
    // The four retired values are what a segmenter trained before the medium was dropped answers.
    // They must keep mapping to their own enum values rather than to the collapsed ones, so a
    // caller reading an old model is never told `series` where the file said `animated_series`.
    CHECK(model::schema::contentKind("live_action_movie") == ContentKind::LiveActionMovie);
    CHECK(model::schema::contentKind("animated_series") == ContentKind::AnimatedSeries);
    CHECK(model::schema::contentKind("movie") == ContentKind::Movie);
    CHECK(model::schema::contentKind("series") == ContentKind::Series);
    CHECK(model::schema::animeKind("True") == AnimeKind::Yes);
    CHECK(model::schema::animeKind("False") == AnimeKind::No);
    CHECK(model::schema::animeKind("") == AnimeKind::Unknown);
}

TEST_CASE("typed learned spans produce the finished value object") {
    const std::string name = "Movie.2024.2160p.UHD.AMZN.WEB-DL.HEVC.10bit.DV.DDP.5.1.Atmos-GROUP.mkv";
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysisOf(name));
    CHECK(info.valid);
    CHECK(info.title == "Movie");
    CHECK(info.year == 2024);
    CHECK(info.screenSize == ResolutionTier::P2160);
    CHECK(info.source == SourceKind::WebDl);
    CHECK(info.streamingService == "AMZN");
    CHECK(info.videoCodec == VideoCodec::Hevc);
    CHECK(info.tenBit);
    CHECK(info.dolbyVision);
    CHECK(info.hdr == "DV");
    CHECK(info.audioCodec == "DDP");
    CHECK(info.audioChannels == "5.1");
    CHECK(info.audioProfile == "Atmos");
    CHECK(info.releaseGroup == "GROUP");
    CHECK(info.container == "mkv");
    CHECK(info.medium == MediumKind::Video);
    CHECK(info.content == ContentKind::Movie);
    CHECK_FALSE(info.anime);
    CHECK(info.animeConfidence == doctest::Approx(0.93F));
    CHECK(info.numbering == NumberingKind::None);
    CHECK(info.packScope == PackScope::Single);
    CHECK(info.special == SpecialKind::Movie);
    CHECK_FALSE(info.specials);
    // 19, not 18: the audio feature span now records its own origin, as every other located
    // fact does. It used to set a boolean and leave no evidence behind.
    CHECK(info.origins.size() == 19);
    const auto profile = std::ranges::find_if(info.origins, [](const FieldOrigin& origin) {
        return origin.field == Field::Audio && origin.value == "Atmos";
    });
    REQUIRE(profile != info.origins.end());
    CHECK(info.audioProfile == "Atmos");
    const auto channels = std::ranges::find_if(info.origins, [](const FieldOrigin& origin) {
        return origin.field == Field::Audio && origin.value == "5.1";
    });
    REQUIRE(channels != info.origins.end());
    CHECK(channels->text == "5.1");
}

TEST_CASE("movie is not a special while ova and special are") {
    const std::string name = "Film";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Film")};

    analysis.specialKind = {"movie", 0.7F};
    CHECK_FALSE(releaseInfoFromAnalysis(name, analysis).specials);
    analysis.specialKind = {"none", 0.7F};
    CHECK_FALSE(releaseInfoFromAnalysis(name, analysis).specials);
    analysis.specialKind = {"ova", 0.7F};
    CHECK(releaseInfoFromAnalysis(name, analysis).specials);
    analysis.specialKind = {"special", 0.7F};
    CHECK(releaseInfoFromAnalysis(name, analysis).specials);

    analysis.specialKind = {"movie", 0.7F};
    analysis.special = SpecialKind::Special;
    CHECK(releaseInfoFromAnalysis(name, analysis).specials);
}

TEST_CASE("coverage verdicts are typed and preserve zero markers") {
    const std::string name = "Show.S00.E01-E03.Complete";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Show"), span(name, SpanType::SeasonMarker, "S00"),
                      span(name, SpanType::EpisodeMarker, "E01-E03"),
                      span(name, SpanType::PackMarker, "Complete")};
    // A SEASON MARKER BESIDE AN ABSOLUTE VERDICT STAYS ABSOLUTE. Two separate spans said season
    // and episode, and anime does exactly this - `Ace of the Diamond ~Second Season~ - 32` pairs a
    // season with series-wide numbering, and the labels read it as absolute. Only a marker that
    // states BOTH numbers in one span can overrule the verdict; see the case below.
    analysis.numberingKind = {"absolute", 0.8F};
    analysis.packScope = {"complete", 0.9F};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.season == 0);
    CHECK(info.absoluteEpisode == 1);
    CHECK(info.absoluteEpisodeEnd == 3);
    CHECK(!info.episode);
    CHECK(!info.episodeEnd);
    CHECK(info.numbering == NumberingKind::Absolute);
    CHECK(info.episodeCount == 3);
    CHECK(info.pack);
    CHECK(info.explicitCompleteRange);
}

TEST_CASE("one span stating both numbers overrules an absolute verdict") {
    // The whole-name head called this series-numbered while the span itself asserted a season and
    // an episode. Converting anyway left season 8 with absolute episode 4 and no episode at all -
    // a season holding nothing. The span wins, and the verdict is corrected with it so the result
    // and its own origins agree.
    const std::string name = "Show Name 804 vostfr HD";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Show Name"),
                      span(name, SpanType::SeasonEpisodeMarker, "804"),
                      span(name, SpanType::SubtitleLanguage, "vostfr")};
    analysis.numberingKind = {"absolute", 0.72F};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.season == 8);
    CHECK(info.episode == 4);
    CHECK(!info.absoluteEpisode);
    CHECK(info.numbering == NumberingKind::SeasonEpisode);
    const auto verdict = std::ranges::find_if(info.origins, [](const FieldOrigin& origin) {
        return origin.field == Field::NumberingKind;
    });
    REQUIRE(verdict != info.origins.end());
    CHECK(verdict->value == "season_episode");   // the origins say what the result says
}

TEST_CASE("a combined marker fills both fields from one span") {
    // `Cap.104` under a stated `Temporada 3`-style release: ONE span the model typed as stating
    // both, so the mapper must produce a season AND an episode from it and record two origins.
    const std::string name = "El Pueblo Cap.301 HDTV";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "El Pueblo"),
                      span(name, SpanType::SeasonEpisodeMarker, "Cap.301"),
                      span(name, SpanType::SourceType, "HDTV")};
    analysis.numberingKind = {"season_episode", 0.9F};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.season == 3);
    CHECK(info.episode == 1);
    CHECK(!info.episodeEnd);
    CHECK_FALSE(info.pack);
    // Two origins, so `stated` and the C ABI see it exactly as they would two separate markers.
    int seasons = 0, episodes = 0;
    for (const FieldOrigin& origin : info.origins) {
        if (origin.field == Field::Season) ++seasons;
        if (origin.field == Field::Episode) ++episodes;
    }
    CHECK(seasons == 1);
    CHECK(episodes == 1);
}

TEST_CASE("a combined marker range shares one season") {
    const std::string name = "El Pueblo Cap.301_302 HDTV";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "El Pueblo"),
                      span(name, SpanType::SeasonEpisodeMarker, "Cap.301_302")};
    analysis.numberingKind = {"season_episode", 0.9F};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.season == 3);
    CHECK(info.episode == 1);
    CHECK(info.episodeEnd == 2);
    CHECK(info.pack);
}

TEST_CASE("season-cross and position-of markers are not episode ranges") {
    const std::string cross = "Show 2x11 HDTV";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(cross, SpanType::Main, "Show"),
                      span(cross, SpanType::EpisodeMarker, "2x11")};
    analysis.numberingKind = {"season_episode", 0.8F};
    ReleaseInfo info = releaseInfoFromAnalysis(cross, analysis);
    CHECK(info.season == 2);
    CHECK(info.episode == 11);
    CHECK(!info.episodeEnd);
    CHECK_FALSE(info.pack);

    const std::string stated = "Show Series 7 04of10";
    analysis.spans = {span(stated, SpanType::Main, "Show"),
                      span(stated, SpanType::SeasonMarker, "Series 7"),
                      span(stated, SpanType::EpisodeMarker, "04of10")};
    info = releaseInfoFromAnalysis(stated, analysis);
    CHECK(info.season == 7);
    CHECK(info.episode == 4);
    CHECK(!info.episodeEnd);
    CHECK_FALSE(info.pack);
}

TEST_CASE("unknown converted spans remain visible as unconverted evidence") {
    const std::string name = "Title.MYSTERY";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Title"), span(name, SpanType::Codec, "MYSTERY")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    REQUIRE(info.origins.size() == 2);
    CHECK(info.origins[1].field == Field::Codec);
    CHECK(info.origins[1].text == "MYSTERY");
    CHECK(info.origins[1].value.empty());
    CHECK(info.origins[1].unconverted);
}

TEST_CASE("invalid analysis returns an explicit invalid result") {
    const ReleaseInfo info = releaseInfoFromAnalysis("Anything", {});
    CHECK_FALSE(info.valid);
    CHECK(info.rawName == "Anything");
    CHECK(info.origins.empty());
}

TEST_CASE("a refused checksum that reads as a 3D layout is a 3D rip") {
    const std::string name = "Example.Movie.2024.3D.Half-SBS.1080p.BluRay.HEVC-GRP";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Example.Movie"), span(name, SpanType::Year, "2024"),
                      span(name, SpanType::Crc32, "3D"), span(name, SpanType::Crc32, "Half-SBS"),
                      span(name, SpanType::Resolution, "1080p"), span(name, SpanType::SourceType, "BluRay"),
                      span(name, SpanType::Codec, "HEVC"), span(name, SpanType::ReleaseGroup, "GRP")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.threeD);
    CHECK(info.screenSize == ResolutionTier::P1080);
    CHECK(originOf(info, Field::Crc32) == nullptr);
    // A real checksum is still a checksum, and never a 3D tag.
    const std::string real = "Show.S01E01.1080p-GRP.[ABCD1234]";
    Analysis checksum;
    checksum.valid = true;
    checksum.spans = {span(real, SpanType::Main, "Show"), span(real, SpanType::Crc32, "ABCD1234")};
    const ReleaseInfo withChecksum = releaseInfoFromAnalysis(real, checksum);
    CHECK(!withChecksum.threeD);
    REQUIRE(originOf(withChecksum, Field::Crc32) != nullptr);
    CHECK(originOf(withChecksum, Field::Crc32)->value == "ABCD1234");
}

TEST_CASE("a bare dub tag is the English dub; a named one is its language") {
    const std::string name = "[Pn8] Date A Live S02 [1080p] [Dub]";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::ReleaseGroup, "Pn8"), span(name, SpanType::Main, "Date A Live"),
                      span(name, SpanType::SeasonMarker, "S02"), span(name, SpanType::Resolution, "1080p"),
                      span(name, SpanType::AudioLanguage, "Dub")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.englishDub);
    CHECK(info.languages.empty());
    REQUIRE(originOf(info, Field::AudioLanguage) != nullptr);
    CHECK(originOf(info, Field::AudioLanguage)->value == "english dub");
    CHECK(!originOf(info, Field::AudioLanguage)->unconverted);

    const std::string german = "Show.S01.German.DL.1080p-GRP";
    Analysis named;
    named.valid = true;
    named.spans = {span(german, SpanType::Main, "Show"), span(german, SpanType::AudioLanguage, "German")};
    const ReleaseInfo withLanguage = releaseInfoFromAnalysis(german, named);
    CHECK(!withLanguage.englishDub);
    CHECK(withLanguage.languages == std::vector<std::string>{"deu"});
}

TEST_CASE("a lower resolution after a higher one across 'to' is a downscale") {
    const std::string name = "Example.Movie.2024.4k.to.1080p.WEBRip.x265-GRP";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Example.Movie"), span(name, SpanType::Year, "2024"),
                      span(name, SpanType::Resolution, "4k"), span(name, SpanType::Resolution, "1080p"),
                      span(name, SpanType::SourceType, "WEBRip"), span(name, SpanType::Codec, "x265")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.downscaled);
    CHECK(info.screenSize == ResolutionTier::P1080);

    // Two resolutions with no connective: the higher wins and nothing is a downscale.
    const std::string plain = "Example.Movie.2024.2160p.1080p.WEBRip-GRP";
    Analysis two;
    two.valid = true;
    two.spans = {span(plain, SpanType::Main, "Example.Movie"), span(plain, SpanType::Resolution, "2160p"),
                 span(plain, SpanType::Resolution, "1080p")};
    const ReleaseInfo higher = releaseInfoFromAnalysis(plain, two);
    CHECK(!higher.downscaled);
    CHECK(higher.screenSize == ResolutionTier::P2160);
}

TEST_CASE("an episode range states its own count") {
    const std::string name = "[SubsPlease] Replica Date, Koi wo Suru. (01-13) (1080p)";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::ReleaseGroup, "SubsPlease"),
                      span(name, SpanType::Main, "Replica Date, Koi wo Suru"),
                      span(name, SpanType::EpisodeMarker, "01-13"), span(name, SpanType::Resolution, "1080p")};
    analysis.numberingKind = {"absolute", 0.9F};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.absoluteEpisode == 1);
    CHECK(info.absoluteEpisodeEnd == 13);
    CHECK(info.episodeCount == 13);
    CHECK(info.pack);
    // Bracketed and ten or more: the fansub batch, complete as far as the uploader had it.
    CHECK(info.explicitCompleteRange);

    // Three episodes in brackets are three episodes; a bare range of thirteen is not a batch.
    const std::string three = "[Group] Show (01-03) (1080p)";
    Analysis short_;
    short_.valid = true;
    short_.spans = {span(three, SpanType::Main, "Show"), span(three, SpanType::EpisodeMarker, "01-03")};
    CHECK(!releaseInfoFromAnalysis(three, short_).explicitCompleteRange);
    const std::string bare = "Show.E01-E13.1080p";
    Analysis unbracketed;
    unbracketed.valid = true;
    unbracketed.spans = {span(bare, SpanType::Main, "Show"), span(bare, SpanType::EpisodeMarker, "E01-E13")};
    CHECK(!releaseInfoFromAnalysis(bare, unbracketed).explicitCompleteRange);
}

TEST_CASE("a dub tag that names its language is the English dub only when it says so") {
    const std::string english = "[Yameii] KOWLOON GENERIC ROMANCE - S01 [English Dub] [CR WEB-DL 1080p]";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(english, SpanType::Main, "KOWLOON GENERIC ROMANCE"),
                      span(english, SpanType::AudioLanguage, "English Dub")};
    const ReleaseInfo info = releaseInfoFromAnalysis(english, analysis);
    CHECK(info.englishDub);
    CHECK(info.languages == std::vector<std::string>{"eng"});

    const std::string german = "Show.S01.1080p.GerDub.WEB-GRP";
    Analysis named;
    named.valid = true;
    named.spans = {span(german, SpanType::Main, "Show"), span(german, SpanType::AudioLanguage, "GerDub")};
    const ReleaseInfo withGerman = releaseInfoFromAnalysis(german, named);
    CHECK(!withGerman.englishDub);
    CHECK(withGerman.languages == std::vector<std::string>{"deu"});
}


// THE THREE READINGS MODEL 3 STOPPED PRODUCING SPANS FOR, each one filled from the name after the
// walk. They were found by an application's own tests, not by this suite: every case here parsed
// correctly under model 2 because the segmenter marked the span, and silently lost the field when
// it stopped. The rules fill only what the model leaves empty, so a later segmenter retires them.

TEST_CASE("a connective swallowed into the resolution span is still a downscale") {
    // `4k.to` as ONE span: the text between the two resolutions is then a bare separator, and the
    // downscale used to read as a mislabel - publishing the resolution the file was scaled FROM.
    const std::string name = "Example.Movie.2024.4k.to.1080p.WEBRip.x265-GRP";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::Main, "Example.Movie"), span(name, SpanType::Year, "2024"),
                      span(name, SpanType::Resolution, "4k.to"),
                      span(name, SpanType::Resolution, "1080p"),
                      span(name, SpanType::SourceType, "WEBRip"), span(name, SpanType::Codec, "x265")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.downscaled);
    CHECK(info.screenSize == ResolutionTier::P1080);
}

TEST_CASE("a 3D layout tag with no span of its own is still a 3D rip") {
    const std::string name = "Example.Movie.2024.3D.Half-SBS.1080p.BluRay.HEVC-GRP";
    Analysis analysis;
    analysis.valid = true;
    // No span for `3D` or `Half-SBS`, which is exactly what the segmenter now returns.
    analysis.spans = {span(name, SpanType::Main, "Example.Movie"), span(name, SpanType::Year, "2024"),
                      span(name, SpanType::Resolution, "1080p"),
                      span(name, SpanType::SourceType, "BluRay"), span(name, SpanType::Codec, "HEVC"),
                      span(name, SpanType::ReleaseGroup, "GRP")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.threeD);
    CHECK(std::ranges::any_of(info.origins, [](const FieldOrigin& origin) {
        return origin.field == Field::ThreeD && origin.begin >= 0;
    }));

    // A TITLE IS NOT A LAYOUT. `3D Kanojo` is a work, and the rule only reads after the title.
    const std::string title = "3D.Kanojo.Real.Girl.S01E01.1080p.WEB-DL-GRP";
    Analysis inTitle;
    inTitle.valid = true;
    inTitle.spans = {span(title, SpanType::Main, "3D.Kanojo.Real.Girl"),
                     span(title, SpanType::Resolution, "1080p")};
    CHECK(!releaseInfoFromAnalysis(title, inTitle).threeD);
}

TEST_CASE("titles with no main reading still publish one") {
    // Two title spans, both typed as alternates: the name used to come back with no title at all,
    // and an application matching a request against the title dropped the release entirely.
    const std::string name = "[Ironclad] Kowloon Generic Romance - S01 [BD.1080p.AV1]";
    Analysis analysis;
    analysis.valid = true;
    analysis.spans = {span(name, SpanType::ReleaseGroup, "Ironclad"),
                      span(name, SpanType::Alternate, "Kowloon Generic Romance"),
                      span(name, SpanType::SeasonMarker, "S01"),
                      span(name, SpanType::Resolution, "1080p")};
    const ReleaseInfo info = releaseInfoFromAnalysis(name, analysis);
    CHECK(info.title == "Kowloon Generic Romance");
    CHECK(info.alternativeTitles.empty());
    CHECK(std::ranges::any_of(info.origins, [](const FieldOrigin& origin) {
        return origin.field == Field::Title && origin.value == "Kowloon Generic Romance";
    }));

    // A STATED MAIN SPAN ALWAYS WINS: the alternate stays an alternate.
    Analysis withMain = analysis;
    withMain.spans[1].type = SpanType::Main;
    withMain.spans.push_back(span(name, SpanType::Alternate, "Kowloon Generic Romance"));
    const ReleaseInfo stated = releaseInfoFromAnalysis(name, withMain);
    CHECK(stated.title == "Kowloon Generic Romance");
    CHECK(stated.alternativeTitles.size() == 1);
}
