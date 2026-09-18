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
    // A RESTORATION IS NOT A REMASTER, and it took three measurements to land on that. Folded
    // into Remastered it cost two validation names; dropped entirely it left GuessIt's
    // `Restored` check unscoreable. Its own kind satisfies both, and says the truer thing:
    // a restoration repairs damaged materials, a remaster re-derives from undamaged ones.
    CHECK(editionIn("Amityville.1992.RESTORED.BDRip") == EditionKind::Restored);
    CHECK(editionIn("Black.Venus.1983.Regraded.German") == EditionKind::Restored);
    CHECK(editionIn("Pelicula.1992.Remasterizado.BDRip") == EditionKind::Remastered);
    // The apostrophe and the `s` are both optional in the wild.
    CHECK(editionIn("Riddick.Unrated.Director.Cut.French") == EditionKind::DirectorsCut);
    // One phrase, two editions - which is why the corpus test reads `editionsIn`, not the
    // first row that matches. `Extended` sits earlier in the table and would hide the other.
    CHECK(editionsIn("Queen.A.Kind.of.Magic.Alternative.Extended.Version")
          == std::vector<EditionKind>{EditionKind::Extended, EditionKind::AlternateCut});
    CHECK(editionIn("Stargate.SG1.Ultimate.Fan.Collection") == EditionKind::Ultimate);
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
    // A BARE ORDINAL IS NOT AN EDITION, and neither is an abbreviation of Anniversary. Both were
    // read as editions for a while and both COST a name on the validation split: `10th.Annv.Ed`
    // was already readable as its own text, and normalising it lost that.
    CHECK(editionIn("Some.Game.10th.Annv.Ed") == EditionKind::Unknown);
    CHECK(editionIn("Movie.21st.Edition.2023") == EditionKind::Numbered);
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

    // QUEUE FILE 02. A screener is the first source the vocabulary was missing outright rather
    // than spelling badly. Only a token saying nothing but `screener` answers it: `DVDScr` keeps
    // the disc, which is the more useful half because it implies a resolution tier.
    CHECK(sourceValue("SCREENER") == SourceKind::Screener);
    CHECK(sourceValue("DVDScr") == SourceKind::Dvd);
    CHECK(sourceValue("WEBSCR") == SourceKind::Screener);
    CHECK(sourceValue("Workprint") == SourceKind::Screener);
    CHECK(sourceValue("BDRip") == SourceKind::BluRay);
    CHECK(sourceValue("ts-hq") == SourceKind::Cam);
    // Misspellings worth tolerating, because the intent is unambiguous in every one.
    CHECK(sourceValue("Bluury") == SourceKind::BluRay);
    CHECK(sourceValue("DVRiP") == SourceKind::Dvd);
    CHECK(sourceValue("UltraHD") == SourceKind::BluRay);
    CHECK(sourceTokenIsBareUhd("Ultra.HD"));
    CHECK(resolutionValue("Movie.2160p.Ultra.HD.BluRay") == ResolutionTier::P2160);
    // `THD+` is what the DD+ rule hands over for `[THD+AC3]`: there the plus is the separator.
    CHECK(audioCodecValue("THD") == "TrueHD");
    CHECK(audioCodecValue("THD+") == "TrueHD");
    CHECK(audioCodecValue("ACC") == "AAC");
    CHECK(audioCodecValue("OGG") == "VORBIS");

    // `Censored` sits before the Uncensored rows, and cannot be reached from inside `UNCENSORED`
    // because the row demands a separator where that word has a letter.
    CHECK(editionIn("Kakyuusei.2.ep2.eng.subs.censored") == EditionKind::Censored);
    CHECK(editionIn("Aneimo.UNCENSORED.1080p") == EditionKind::Uncensored);
    CHECK(editionIn("Movie.Decensored.1080p") == EditionKind::Uncensored);
    CHECK(editionIn("Star.Trek.TOS.s01e11e12.FANEDIT.900p") == EditionKind::FanEdit);
    CHECK(editionIn("System_Of_A_Down-Leeds-Bootleg-2001-STA") == EditionKind::Bootleg);
    CHECK(editionIn("Cross_Game_1-50_unofficial-batch") == EditionKind::Unofficial);
    // The real span is the word alone; spelled out here because `UNCUT` sits earlier in the
    // table and `editionIn` answers with the first row that matches, not the best one.
    CHECK(editionIn("Moontrap.1989.BONUS.GERMAN.DVD9") == EditionKind::Bonus);
    CHECK(editionsIn("Moontrap.1989.BONUS.UNCUT.GERMAN")
          == std::vector<EditionKind>{EditionKind::Uncut, EditionKind::Bonus});
    CHECK(editionIn("Farmhouse.2008.FESTiVAL.DVDRip") == EditionKind::Festival);
    CHECK(editionIn("Mathilde.2004.2DISC.GERMAN.DVD9") == EditionKind::MultiDisc);
    // A single disc is not a multi-disc release, which is why the count starts at two.
    CHECK(editionIn("Mathilde.2004.1DISC.GERMAN.DVD9") == EditionKind::Unknown);
    CHECK(editionIn("Castle.in.the.Sky.1986.RM.1080p") == EditionKind::Remastered);
    CHECK(editionIn("Berserk.MEMORIAL.EDITION.02") == EditionKind::SpecialEdition);

    // QUEUE FILE 03. RealVideo is a codec the enum lacked; `.rmvb` releases still carry it.
    CHECK(codecValue("RV10") == VideoCodec::RealVideo);
    CHECK(codecValue("RV40") == VideoCodec::RealVideo);
    CHECK(codecValue("RV") == VideoCodec::Unknown);
    // A platform with `DL` glued on is the mirror of the platform-rip rule: nothing says `WEB`.
    CHECK(sourceValue("CR-DL") == SourceKind::WebDl);
    CHECK(sourceValue("LDRip") == SourceKind::Dvd);
    CHECK(sourceValue("蓝光") == SourceKind::BluRay);
    CHECK(audioCodecValue("WAV") == "PCM");

    CHECK(editionIn("Movie.2024.LEAKED.1080p") == EditionKind::Leaked);
    CHECK(editionIn("Rewind.1990s.S01E04.SHORTENED.720p") == EditionKind::Shortened);
    CHECK(editionIn("Puppet.Master.2003.ALTERNATIVE.CUT.720P") == EditionKind::AlternateCut);
    CHECK(editionIn("Phaeton.an.Erde.1981.Alternate.Cut.German") == EditionKind::AlternateCut);
    CHECK(editionIn("Arrested.Development.S04.Remix.Part.1") == EditionKind::AlternateCut);
    CHECK(editionIn("Holiday.Inn.1942.Colorized.1080p") == EditionKind::Colorized);
    CHECK(editionIn("The.Running.Man.1987.FS.DVDRip") == EditionKind::Fullscreen);
    CHECK(editionIn("Movie.1987.WS.DVDRip") == EditionKind::Widescreen);
    CHECK(editionIn("Human.Condition.1959.CC.1080p") == EditionKind::Criterion);
    CHECK(editionIn("The.Faculty.1998.SHOUT.CE.1080p") == EditionKind::Collector);
    CHECK(editionIn("Hamidashi.Creative.豪華版") == EditionKind::Deluxe);
    CHECK(editionIn("Culture.Club.Kissing.To.Be.Clever.Expanded.Edition") == EditionKind::Extended);
    // PROPERFIX runs two words together, so both rules used to miss it: it is a proper AND a fix.
    CHECK(editionIn("Arabasta.17.PROPERFIX-One.Pace") == EditionKind::Fix);
    // A bare PROPER stays a proper - it is not an edition of its own.
    CHECK(editionIn("Movie.2024.PROPER.1080p") == EditionKind::Unknown);
    CHECK(languageCodesOfToken("国英双语") == std::vector<std::string>{"cmn", "eng"});
    CHECK(languageCodesOfToken("简繁日双语") == std::vector<std::string>{"zho", "jpn"});

    // QUEUE FILE 04. `UHD2BD` states both ends of a re-encode; the target is what arrives.
    CHECK(sourceValue("UHD2BD") == SourceKind::BluRay);
    CHECK(sourceValue("BD9") == SourceKind::BluRay);
    CHECK(sourceValue("DLrip") == SourceKind::WebRip);
    CHECK(sourceValue("FBRip") == SourceKind::WebRip);
    CHECK(sourceValue("VoDHD") == SourceKind::WebDl);
    CHECK(audioCodecValue("WMA") == "WMA");
    CHECK(audioCodecValue("APE") == "APE");
    CHECK(containerValue("F4V") == "f4v");

    CHECK(editionIn("Blu-ray.BOX.通常版.240228") == EditionKind::Standard);
    CHECK(editionIn("Grenadier.OP.01-02.Creditless.DVD") == EditionKind::Creditless);
    CHECK(editionIn("Show.S01E01.NCOP1.1080p") == EditionKind::Creditless);
    CHECK(editionIn("Taylor.Swift.The.Eras.Tour.2023.Taylors.Version") == EditionKind::ReRecorded);
    CHECK(editionIn("Dragon.Ball.Path.To.Power.Edited.DVD") == EditionKind::AlternateCut);
    CHECK(editionIn("Star.Trek.1979.The.Directors.Edition.German") == EditionKind::DirectorsCut);
    CHECK(editionIn("The.Magicians.S04E01.PROOF.BDRip") == EditionKind::Fix);
    CHECK(editionIn("DOOM.I.and.II.Enhanced.Repack") == EditionKind::Remastered);
    CHECK(editionIn("King.Solomons.Mines.1985.DEU.Transfer.BDRip") == EditionKind::Remastered);
    CHECK(editionIn("Watch.Dogs.2.Gold.Edition.Repack") == EditionKind::SpecialEdition);
    // AN AI UPSCALE IS NOT A REMASTER. Without the lookbehind on the Enhanced row, the edition
    // label wins over the AI-upscale routing and the fact is buried - which it was, once.
    CHECK(editionIn("Predator.2.1990.2160p.Ai-Enhanced.HEVC") == EditionKind::Unknown);
    CHECK(editionIn("Movie.2024.AI.Enhanced.1080p") == EditionKind::Unknown);
    // `年齡限制版` is the version that KEEPS the material and carries the rating, not one cut for it.
    CHECK(editionIn("[ANi] NUKITASHI [年齡限制版] - 04") == EditionKind::Uncut);
    CHECK(languageCodesOfToken("国粤语音轨") == std::vector<std::string>{"cmn", "yue"});
    CHECK(languageCodesOfToken("简／繁") == std::vector<std::string>{"zho"});

    // QUEUE FILE 05. A Digital Cinema Package is neither a disc nor a stream, and answering one
    // of those would have been the nearest wrong answer rather than the right one.
    CHECK(sourceValue("DCP") == SourceKind::DigitalCinema);
    CHECK(sourceValue("DCPRip") == SourceKind::DigitalCinema);
    CHECK(sourceValue("R5") == SourceKind::Dvd);
    CHECK(sourceValue("AVCHD") == SourceKind::BluRay);
    CHECK(sourceValue("BDISO") == SourceKind::BluRay);
    CHECK(sourceValue("BRDRip") == SourceKind::BluRay);
    CHECK(sourceValue("ブルーレイディスク") == SourceKind::BluRay);

    CHECK(editionIn("Jury.Duty.Cast.Commentary.Edition.S01E01") == EditionKind::Commentary);
    CHECK(editionIn("The.Getaway.1994.Explicit.1080p.BluRay") == EditionKind::Explicit);
    CHECK(editionIn("Chirco.Visitation.Reissue.1972") == EditionKind::Reissue);
    CHECK(editionIn("From.Beyond.1986.UC.German.HDRip") == EditionKind::Uncut);
    CHECK(editionIn("LOTR.II.2002.EXT.BDRemux") == EditionKind::Extended);
    // `EXTRAS` is not `EXT`: the abbreviation has to end the token.
    CHECK(editionIn("Show.S01.EXTRAS.1080p") == EditionKind::Unknown);
    CHECK(editionIn("Karami.Zakari.カラー化.zip") == EditionKind::Colorized);
    CHECK(editionIn("[DBD-Raws][约会大作战][导演剪辑版][1080P]") == EditionKind::DirectorsCut);
    // A director's COMMENTARY is not a director's CUT, and the cut row is asked first.
    CHECK(editionIn("Movie.2004.Directors.Commentary.1080p") == EditionKind::Commentary);
    CHECK(languageCodesOfToken("简繁字幕外挂") == std::vector<std::string>{"zho"});

    // QUEUE FILE 06. A scanned print is not a telecine: a scene TELECINE is a leak, `35MM.FilmScan`
    // is someone's own scan of their own reel, and the cam branch is asked first.
    CHECK(sourceValue("35MM") == SourceKind::Film);
    CHECK(sourceValue("35MM.FilmScan") == SourceKind::Film);
    CHECK(sourceValue("TELECINE") == SourceKind::Cam);
    CHECK(sourceValue("IPTV") == SourceKind::Hdtv);
    CHECK(sourceValue("FEED") == SourceKind::Hdtv);
    CHECK(sourceValue("LDTV") == SourceKind::Hdtv);
    CHECK(sourceValue("StreamRip") == SourceKind::WebRip);
    // `Hybrid` names no source ON PURPOSE - it says two were combined and refuses to say which.
    CHECK(sourceValue("Hybrid") == SourceKind::Unknown);

    // OAR is not Widescreen: it says the frame was not reframed, whatever shape that frame is.
    CHECK(editionIn("Knight.Rider.2000.1991.OAR.BDRIP") == EditionKind::OriginalAspectRatio);
    CHECK(editionIn("Movie.1989.Arrow.1080p.BluRay") == EditionKind::Remastered);
    CHECK(editionIn("The.Drummer.2015.RE-EDIT.BDRip") == EditionKind::AlternateCut);
    CHECK(editionIn("Garden.(別版).zip") == EditionKind::AlternateCut);
    CHECK(editionIn("Udo.Lindenberg.2012.EXTRA.GERMAN.MBluRay") == EditionKind::Bonus);
    CHECK(editionIn("【フルカラー版】Doujin.zip") == EditionKind::Colorized);
    CHECK(editionIn("Doujin.【デジタル特装版】.zip") == EditionKind::SpecialEdition);
    CHECK(editionIn("El jugador (ilustrado) [51149].epub") == EditionKind::SpecialEdition);
    CHECK(languageCodesOfToken("简繁英双语字幕") == std::vector<std::string>{"zho", "eng"});
    CHECK(languageCodesOfToken("简繁日字幕") == std::vector<std::string>{"zho", "jpn"});

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

    // A RANGE ANSWERS ITS START. One span stating a span of TIME is not several years stated
    // separately, and the scan that walks a whole name keeps the last year it finds - right for
    // the second case and wrong for the first. Measured on the double-year set, this and the
    // connectives below took the conversion layer from 92.7% to 99.7% on spans the model and the
    // gold both located.
    CHECK(yearIn("Eyes.On.The.Prize.1954-1956.Fighting.Back").value == "1954");
    CHECK(yearIn("The Twilight Zone - Season 1 - 1958 thru 1960").value == "1958");
    CHECK(yearIn("As Aventuras de Paddington 2014 e 2018").value == "2014");
    CHECK(yearIn("Batman Heptalogy (1989 to 2017)").value == "1989");
    // Two years stated SEPARATELY still answer the last, which is the release year beside a year
    // that belongs to the work's own text.
    CHECK(yearIn("Show.2019.The.Great.Air.Race.of.1924").value == "1924");
    const Reading year = yearIn("Movie.1920x1080.2019");
    CHECK(year.value == "2019");
    CHECK(yearIn("Show.2019.2023.06.01", 10, 20).value == "2019");
}
