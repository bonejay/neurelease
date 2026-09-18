#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "convert/field_values.hpp"

using namespace neurelease;
using namespace neurelease::convert;

TEST_CASE("resolution preserves evidence and uses byte offsets") {
    CHECK(resolutionValue("1920x816") == ResolutionTier::P1080);

    CHECK(resolutionValue("1O8Op") == ResolutionTier::P1080);
    CHECK(resolutionValue("4096p") == ResolutionTier::P2160);
    CHECK(resolutionValue("Movie.1080.Days") == ResolutionTier::P1080);
    CHECK(resolutionValue("电影.1080p") == ResolutionTier::P1080);
}

TEST_CASE("video vocabulary stays canonical and typed") {
    CHECK(sourceValue("[BD1080p]") == SourceKind::BluRay);
    CHECK(sourceValue("AMZN.WEB-DL") == SourceKind::WebDl);
    CHECK(sourceValue("HDTS") == SourceKind::Cam);
    CHECK(sourceValue("mystery") == SourceKind::Unknown);
    CHECK(sourceTokenIsRemux("BDRemux"));
    CHECK(sourceTokenIsLightEncode("MicroHD"));
    CHECK(sourceTokenIsBareUhd("UHD"));

    // A source-position token that states other fields too. Narrow on purpose: `UHDRDV` is one
    // spelling with one meaning, and a bare `UHD` must keep the tentative reading above rather
    // than being swallowed by a substring rule.
    CHECK(sourceTokenExtras("UHDRDV").screenSize == ResolutionTier::P2160);
    CHECK(sourceTokenExtras("UHDRDV").hdr10);
    CHECK(sourceTokenExtras("UHDRDV").dolbyVision);
    CHECK(sourceTokenExtras("UHDR").hdr10);
    CHECK_FALSE(sourceTokenExtras("UHDR").dolbyVision);
    CHECK_FALSE(sourceTokenExtras("UHD").any());
    CHECK_FALSE(sourceTokenExtras("BluRay").any());

    // QUEUE 01 of the unmapped-span triage. `FINAL` and `Final Cut` are the pair worth pinning:
    // French releases write `S01E08.FiNAL` for a season's last episode 3,133 times, and sending
    // that to the director's final cut would be wrong on every one.
    CHECK(editionIn("1983.S01E08.FiNAL.FRENCH") == EditionKind::Final);
    CHECK(editionIn("Blade.Runner.1982.Final.Cut") == EditionKind::FinalCut);
    CHECK(editionIn("Childs.Play.PROOFFIX") == EditionKind::Fix);
    CHECK(editionIn("Show.1080p-fixed") == EditionKind::Fix);
    CHECK(editionIn("Wild.World.Complete.Edition") == EditionKind::CompleteEdition);
    CHECK(editionIn("Days.of.High.Adventure.Unabridged") == EditionKind::Unabridged);
    CHECK(editionIn("Roller_Coaster-CONVERT-DVDRip") == EditionKind::Reencode);
    CHECK(editionIn("Darby.O.Gill.1959.iNT.DVDRip") == EditionKind::Internal);
    CHECK(editionIn("Aneimo.UNC.1080p") == EditionKind::Uncensored);
    // A restoration, a regrade and a remaster are one fact under three names.
    CHECK(editionIn("Amityville.1992.RESTORED.BDRip") == EditionKind::Remastered);
    CHECK(editionIn("Black.Venus.1983.Regraded.German") == EditionKind::Remastered);
    // The long cut and the cinema cut, named in the language that released them.
    CHECK(editionIn("Man-Eater.1980.Langfassung.German") == EditionKind::Extended);
    CHECK(editionIn("F.I.S.T.1978.KiNOFASSUNG.German") == EditionKind::Theatrical);

    // QUEUE 01, SECOND PASS. UHDRDV and UHDR now answer the source question too, and answer it
    // tentatively - `sourceTokenIsBareUhd` is what lets a WEB-DL later in the name overrule them.
    CHECK(sourceValue("UHDRDV") == SourceKind::BluRay);
    CHECK(sourceValue("UHDR") == SourceKind::BluRay);
    CHECK(sourceTokenIsBareUhd("UHDRDV"));
    // A platform with `RIP` glued on has no `WEB` in it, so every branch used to miss.
    CHECK(sourceValue("NetflixRip") == SourceKind::WebRip);
    CHECK(sourceValue("AMZNRip") == SourceKind::WebRip);
    CHECK(sourceValue("DVDRip") == SourceKind::Dvd);

    // `Numbered` and `Regional` each drop a detail the vocabulary has no field for - which number,
    // which region - and keep the only part a consumer can act on: that one was stated at all.
    CHECK(editionIn("Modern C (MEAP v4) 3ed 2024") == EditionKind::Numbered);
    // The abbreviation beats the Numbered rule, because Anniversary says strictly more.
    CHECK(editionIn("Some.Game.10th.Annv.Ed") == EditionKind::Anniversary);
    CHECK(editionIn("Some.Game.Anniv.Edition") == EditionKind::Anniversary);
    CHECK(editionIn("Lean Six Sigma 2nd Edition 2023") == EditionKind::Numbered);
    CHECK(editionIn("[DBD-Raws][屍鬼][美版][1080P]") == EditionKind::Regional);
    CHECK(editionIn("[DBD-Raws][屍鬼][USA.Ver][1080P]") == EditionKind::Regional);
    // The Chinese encode editions: high bitrate, sixty frames, and the audio equivalent.
    CHECK(editionIn("1921[高码版][国语配音]") == EditionKind::HighQuality);
    CHECK(editionIn("Endless.Journey[60帧率版本].2160p") == EditionKind::HighQuality);
    CHECK(editionIn("[FLAC-tan] (Hi-RES) Clear Card OP") == EditionKind::HighQuality);
    CHECK(editionIn("Cyberpunk.2077-CODEX [Ultimate Edition]") == EditionKind::Ultimate);
    // Decensoring, in the three scripts that state it, and a first-press Japanese retail edition.
    CHECK(editionIn("STARS-160 无码流出") == EditionKind::Uncensored);
    CHECK(editionIn("Kasugano 無碼流出 part 3") == EditionKind::Uncensored);
    CHECK(editionIn("JUX-174 モザイク破壊版") == EditionKind::Uncensored);
    CHECK(editionIn("[BDMV] TAILENDERS【初回限定版】") == EditionKind::Limited);
    // A marketing name for the long cut, and a corrective re-release that names no version.
    CHECK(editionIn("Deadpool.2.2018.Super.Duper.Cut.UNRATED") == EditionKind::Extended);
    CHECK(editionIn("WITCH.Season.1.DVDRip.v2.UPDATED") == EditionKind::Fix);
    CHECK(editionIn("2024_Marie01(remake).mp4") == EditionKind::Reencode);
    // `4К` with a Cyrillic К used to fall through to the 1080p default - a WRONG answer, not an
    // absent one, which is the reason it is pinned here rather than left to the queue.
    CHECK(resolutionValue("Дом Гиннесса _1 сезон_4К_RHS_") == ResolutionTier::P2160);
    CHECK(resolutionValue("Movie_4K_RHS_") == ResolutionTier::P2160);
    CHECK(languageCodesOfToken("官方中字") == std::vector<std::string>{"zho"});

    // Languages added from the same triage. The CJK entries must match the WHOLE token, because
    // the matcher demands an ASCII boundary on each side.
    CHECK(languageCodesOfToken("粤语音轨") == std::vector<std::string>{"yue"});
    CHECK(languageCodesOfToken("swissgerman") == std::vector<std::string>{"deu"});
    CHECK(languageCodesOfToken("vietsub") == std::vector<std::string>{"vie"});
    CHECK(audioCodecValue("H264.MP2") == "MP2");

    CHECK(codecValue("HEVC-10bit") == VideoCodec::Hevc);
    CHECK(codecValue("Hi10P") == VideoCodec::H264);
    CHECK(codecValue("VVC") == VideoCodec::Vvc);
    CHECK(bitDepthIn("Hi10P").value == "10bit");

    CHECK(hdrValue("2160p.DV.HDR10").format == HdrFormat::DolbyVision);
    CHECK(hdrValue("UHDR10+").format == HdrFormat::Hdr10Plus);
    CHECK(hdrFormatLabel(HdrFormat::Hdr10Plus) == "HDR10+");
}

TEST_CASE("editions and audio retain all independent claims") {
    const auto editions = editionsIn("Movie.IMAX.Remastered.Uncut.DC");
    REQUIRE(editions.size() == 4);
    CHECK(editions[0] == EditionKind::Imax);
    CHECK(editions[1] == EditionKind::Remastered);
    CHECK(editions[2] == EditionKind::Uncut);
    CHECK(editions[3] == EditionKind::DirectorsCut);

    // The eight later editions, including the two the model rarely routes here as editions at all
    // (`2in1` reads as a pack marker in most names, `Preair` as an edition) - the conversion still
    // has to answer for them, and this is where that is pinned.
    CHECK(editionIn("Movie.Despecialized.1080p") == EditionKind::Despecialized);
    CHECK(editionIn("Movie.Assembly.Cut.1080p") == EditionKind::AssemblyCut);
    CHECK(editionIn("Movie.25th.Anniversary.Edition") == EditionKind::Anniversary);
    CHECK(editionIn("Movie.Anniversary.Edition") == EditionKind::Anniversary);
    CHECK(editionIn("Movie.Signature.Edition") == EditionKind::Signature);
    CHECK(editionIn("Movie.Imperial.Edition") == EditionKind::Imperial);
    CHECK(editionIn("Movie.Diamond.Edition") == EditionKind::Diamond);
    CHECK(editionIn("Show.S01E01E02.2in1.720p") == EditionKind::TwoInOne);
    CHECK(editionIn("Show.S01E01.PRE-AIR.720p") == EditionKind::Preair);
    CHECK(editionIn("Show.S01E01.PREAiR.720p") == EditionKind::Preair);
    // A bare `Anniversary` is an ordinary title word, and `2in1080p` is a resolution.
    CHECK(editionIn("Anniversary.2023.1080p.WEB-DL") == EditionKind::Unknown);
    CHECK(editionIn("Movie.2in1080p.WEB-DL") == EditionKind::Unknown);

    CHECK(audioCodecValue("DDP5.1") == "DDP");
    CHECK(audioCodecValue("2xFLAC") == "FLAC");
    CHECK(bitDepthIn("Hevc10").value == "10bit");
    CHECK(bitDepthIn("x265-10bit").value == "10bit");
    CHECK(bitDepthIn("HEVC").value.empty());
    CHECK(bitDepthIn("x264").value.empty());
    CHECK(audioProfileValue("Atmos") == "Atmos");
    CHECK(audioProfileValue("ATMOS") == "Atmos");
    CHECK(audioProfileValue("DTS-X") == "DTS:X");
    CHECK(audioProfileValue("DTS:X") == "DTS:X");
    CHECK(audioProfileValue("DTSX") == "DTS:X");
    CHECK(audioProfileValue("Auro-3D") == "Auro-3D");
    CHECK(audioProfileValue("7.1").empty());
    CHECK(audioChannelsValue("6CH") == "5.1");
    CHECK(audioChannelsValue("2CH") == "2.0");
    CHECK(audioChannelsValue("8ch") == "7.1");
    CHECK(audioChannelsValue("7.1CH") == "7.1");
    CHECK(audioChannelsValue("5 1") == "5.1");
}

TEST_CASE("coverage, containers, and language decorations are value conversions") {
    const MarkerNumbers range = markerNumbersIn("Season.1.3.5");
    CHECK(range.stated);
    CHECK(range.first == 1);
    CHECK(range.last == 5);
    CHECK(range.count == 3);
    const MarkerNumbers zero = markerNumbersIn("S00");
    CHECK(zero.stated);
    CHECK(zero.first == 0);

    // `2x11` is season 2 episode 11, never the range 2 to 11, and `04of10` is episode 4 of ten,
    // never the range 4 to 10. A frame size is not a marker: the left operand stays two digits.
    const EpisodeMarkerReading plain = episodeMarkerIn("2x11");
    CHECK(plain.stated);
    CHECK(plain.season == 2);
    CHECK(plain.first == 11);
    CHECK(plain.last == 0);
    CHECK(episodeMarkerIn("1920x1080").season == 0);
    // The episode word may lead the cross, the cross may be the multiplication sign, and a
    // season letter left inside an episode span still says which number is which. All three
    // were GuessIt-corpus spans the model typed correctly and this read as one bare number.
    const EpisodeMarkerReading worded = episodeMarkerIn("Ep 2x03");
    CHECK(worded.season == 2);
    CHECK(worded.first == 3);
    const EpisodeMarkerReading times = episodeMarkerIn("2\xC3\x97" "7");  // 2×7, UTF-8
    CHECK(times.season == 2);
    CHECK(times.first == 7);
    const EpisodeMarkerReading lettered = episodeMarkerIn("s8e6");
    CHECK(lettered.season == 8);
    CHECK(lettered.first == 6);
    const EpisodeMarkerReading spanish = episodeMarkerIn("T01XE08");
    CHECK(spanish.season == 1);
    CHECK(spanish.first == 8);
    const EpisodeMarkerReading position = episodeMarkerIn("04of10");
    CHECK(position.first == 4);
    CHECK(position.last == 0);
    CHECK(position.season == 0);
    // Anything else is still read as the numbers it names.
    const EpisodeMarkerReading listed = episodeMarkerIn("E01-E03");
    CHECK(listed.first == 1);
    CHECK(listed.last == 3);

    // THE COMBINED MARKER. Read only once the model has typed the span, because `104` and the
    // `102` of an anime at episode 102 are the same three digits.
    const EpisodeMarkerReading combined = seasonEpisodeMarkerIn("Cap.104");
    CHECK(combined.stated);
    CHECK(combined.season == 1);
    CHECK(combined.first == 4);
    CHECK(combined.last == 0);
    CHECK(seasonEpisodeMarkerIn("Cap.604").season == 6);
    CHECK(seasonEpisodeMarkerIn("Cap.604").first == 4);
    CHECK(seasonEpisodeMarkerIn("Cap.1503").season == 15);
    CHECK(seasonEpisodeMarkerIn("Cap.1503").first == 3);
    CHECK(seasonEpisodeMarkerIn("1216").season == 12);
    CHECK(seasonEpisodeMarkerIn("1216").first == 16);
    // A range shares one season: `Cap.101_108` is season 1, episodes 1 to 8.
    const EpisodeMarkerReading spread = seasonEpisodeMarkerIn("Cap.101_108");
    CHECK(spread.season == 1);
    CHECK(spread.first == 1);
    CHECK(spread.last == 8);
    CHECK(spread.count == 8);
    const EpisodeMarkerReading wide = seasonEpisodeMarkerIn("Caps.201_210");
    CHECK(wide.season == 2);
    CHECK(wide.first == 1);
    CHECK(wide.last == 10);

    CHECK(containerValue("archive.Movie.MKV") == "mkv");
    CHECK(mediumOfContainer("epub") == MediumKind::Book);
    CHECK(mediumOfContainer("m4b") == MediumKind::Audiobook);
    CHECK(statesAudioSampleDepth("24Bit-96kHz"));
    CHECK(statesSoftwareBitness("x64"));

    CHECK(languageCodesOfToken("EngSub") == std::vector<std::string>{"eng"});
    CHECK(languageCodesOfToken("ENSUB+PLSUB") == std::vector<std::string>{"eng", "pol"});
    CHECK(subtitleFormatValue("ASS") == SubtitleFormat::Ass);
    CHECK(statesSubtitlesOnly("INC SUBS"));
    CHECK(statesSeveralSubtitles("SRTx2"));
}

TEST_CASE("a doubled word is a title, not a duplicated title") {
    // REGRESSION, found on GuessIt's own corpus 2026-09-01. `[DeadFish] Tari Tari - 01` parsed
    // with the span `Tari Tari` and reported the title `Tari`: the duplicate-title collapse fired
    // on a two-word title whose halves happen to match. The span was right the whole time, so
    // nothing in the model or the span tests could see it - only the value was wrong.
    CHECK(titleText("Tari Tari") == "Tari Tari");
    CHECK(titleText("Duran Duran") == "Duran Duran");
    CHECK(titleText("Sing Sing") == "Sing Sing");
    CHECK(titleText("Boum Boum") == "Boum Boum");
    // Still collapsed, which is what the rule is for: a release that writes the title twice.
    CHECK(titleText("Some.Movie.Some.Movie") == "Some Movie");
    CHECK(titleText("The.Big.Show.The.Big.Show") == "The Big Show");
}

TEST_CASE("identity conversion keeps open values while removing syntax") {
    CHECK(titleText("Some.Movie.Some.Movie") == "Some Movie");
    CHECK(preferLatinTitle("羊毛战记 第一季 Silo") == "Silo");
    CHECK(preferLatinTitle("Закулисье реальности / Backrooms") == "Backrooms");
    CHECK(preferLatinTitle("羊毛战记") == "羊毛战记");
    CHECK(groupText("[-GROUP.mkv]") == "GROUP");
    CHECK(groupText("1234").empty());
    CHECK(isNeverAGroup("x265"));
}

TEST_CASE("platform and calendar values are deterministic") {
    CHECK(platformValue("Amazon") == "AMZN");
    CHECK(platformValue("B Global") == "B-GLOBAL");
    CHECK(platformValue("iP") == "iP");

    const DateReading full = dateIn("Show.2019.2023.06.01");
    CHECK(full.date == Date{2023, 6, 1});
    CHECK(full.reading.value == "2023-06-01");
    CHECK(dateIn("Daily.24-02-15").date == Date{2024, 2, 15});
    CHECK_FALSE(dateIn("Episode.01.02.03").date.valid());

    const Reading year = yearIn("Movie.1920x1080.2019");
    CHECK(year.value == "2019");
    CHECK(yearIn("Show.2019.2023.06.01", 10, 20).value == "2019");
}
