"""Every field the binding promises, checked against a live parse.

The smoke test proves the happy path; this file exists because the binding silently dropped two
whole ABI features for weeks — air dates and verdict confidences were carried by the C ABI and
never surfaced — and nothing failed. Each dataclass field is asserted at least once here, so a
field the copy loop forgets breaks a test instead of quietly returning nothing forever.

Run without arguments beside a built library: discovery is part of what is under test, so no
NEURELEASE_MODELS/NEURELEASE_LIBRARY is required when the repo layout provides them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neurelease import (AdultKind, AirDate, ContentKind, EditionKind, MediumKind,
                           NumberingKind, Origin, OriginField, PackScope, ParsedRelease, Parser,
                           NeureleaseError, ResolutionTier, SourceKind, SpecialKind,
                           SubtitleFormat, VideoCodec)


@pytest.fixture(scope="module")
def parser() -> Parser:
    with Parser() as instance:
        yield instance


# --- every dataclass field, asserted once ---------------------------------------------------------

def test_a_full_name_fills_the_flat_fields(parser: Parser) -> None:
    release = parser.parse(
        "Movie.2024.2160p.UHD.AMZN.WEB-DL.DDP5.1.Atmos.DV.10bit.HEVC.EXTENDED-GRP.mkv")
    assert release.valid and not release.degraded
    assert release.title == "Movie"
    assert release.year == 2024
    assert release.screen_size == ResolutionTier.P2160 and release.screen_size.kind == ResolutionTier.P2160
    assert release.source == SourceKind.WEB_DL and release.source.kind == SourceKind.WEB_DL
    assert release.video_codec == VideoCodec.HEVC and release.video_codec.kind == VideoCodec.HEVC
    assert release.medium == MediumKind.VIDEO and release.medium.kind == MediumKind.VIDEO
    assert release.streaming_service == "AMZN"
    assert release.audio_codec == "DDP" and release.audio_channels == "5.1"
    assert release.audio_profile == "Atmos"
    assert release.hdr == "DV"
    assert release.container == "mkv"
    assert release.release_group == ("GRP",)
    assert EditionKind.EXTENDED in release.edition and release.edition[0].kind == EditionKind.EXTENDED
    # A name whose whole title is `Movie` is a movie; live-action against animated is a model
    # judgement on a synthetic name and has flipped between weights, so it is not asserted.
    assert release.content.name.endswith("MOVIE")
    assert release.numbering == NumberingKind.NONE
    assert release.pack_scope == PackScope.SINGLE
    assert release.special == SpecialKind.MOVIE
    assert release.adult == AdultKind.NO
    assert release.tokens > 0


def test_numbering_fields(parser: Parser) -> None:
    ranged = parser.parse("Show.S01E01-E09.1080p.BluRay.x264-GRP")
    assert (ranged.season, ranged.episode, ranged.episode_end) == (1, 1, 9)
    assert ranged.pack
    seasons = parser.parse("Show.S01-S03.COMPLETE.1080p.WEB-DL.x264-GRP")
    assert (seasons.season, seasons.season_end) == (1, 3)
    assert seasons.pack_scope in (PackScope.MULTI_SEASON, PackScope.COMPLETE)
    absolute = parser.parse("[Group] Show - 728 [1080p].mkv")
    assert absolute.absolute_episode == 728
    assert absolute.numbering == NumberingKind.ABSOLUTE
    specials = parser.parse("Show.S00E01.Special.1080p.WEB-DL.x264-GRP.mkv")
    # THE case the None convention exists for: S00 is a stated season of zero, not an absence.
    assert specials.season == 0 and specials.episode == 1 and specials.specials
    movie = parser.parse("Movie.2024.1080p.WEB-DL.x264-GRP.mkv")
    assert movie.season is None and movie.episode is None  # absent, not zero


def test_property_flags_cross_the_boundary(parser: Parser) -> None:
    dl = parser.parse("Die.Discounter.S04E03.German.DL.2160p.WEB.h265-GRP")
    assert dl.dual_audio                     # only visible in origins before
    remux = parser.parse("Movie.2021.2160p.UHD.BluRay.REMUX.DV.HDR10.TrueHD.Atmos.10bit.H264-GRP")
    assert remux.remux and remux.ten_bit and remux.dolby_vision and remux.hdr10
    plain = parser.parse("Movie.2024.1080p.WEB-DL.x264-GRP.mkv")
    assert not any((plain.remux, plain.dual_audio, plain.multi_subs, plain.hard_subs,
                    plain.proper, plain.repack, plain.dolby_vision))


@pytest.mark.model
def test_multi_read_as_audio_not_subtitles(parser: Parser) -> None:
    """A movie-shaped MULTI is an audio fact - dual or multi audio - and not a subtitle tag.

    WEIGHT-SENSITIVE. The 2026-08-31 weights read this token as a subtitle tag and neither flag
    was set; the 2026-09-03 weights read it as a dual-audio tag. Either audio flag satisfies the
    intent; a subtitle reading does not.
    """
    multi = parser.parse("Turbulence.2025.MULTI.1080p.WEB.H264-BAWLS")
    assert multi.multi_audio or multi.dual_audio
    assert not multi.subtitle_language


@pytest.mark.model
def test_a_work_subtitle_is_inside_the_title(parser: Parser) -> None:
    """Since 2026-09-03 the model reads a work's subtitle as part of its title; there is no
    separate field. WEIGHT-SENSITIVE: the boundary is the model's."""
    release = parser.parse("[Erai-raws] Kimetsu no Yaiba - Hashira Geiko-hen - 01 [1080p].mkv")
    assert release.title == "Kimetsu no Yaiba - Hashira Geiko-hen"


@pytest.mark.model
def test_parenthesised_subtitle_survives_in_the_title_fields(parser: Parser) -> None:
    release = parser.parse("(KRTM)_-_Consumer_(The_Worst_Of_KRTM)-(PRSPCTLP013)-WEB-2018-SRG")
    titles = [release.title, *release.alternative_title] if hasattr(release, "alternative_title") else [release.title]
    assert any("The Worst Of KRTM" in (title or "") for title in titles)


def test_air_date_present_and_absent(parser: Parser) -> None:
    dated = parser.parse("Newsroom Tokyo 2026 08 25 1080p HDTV H264-DARKFLiX")
    assert dated.date == AirDate(year=2026, month=8, day=25)
    assert dated.numbering == NumberingKind.DATE
    undated = parser.parse("Movie.2024.1080p.WEB-DL.x264-GRP.mkv")
    assert undated.date is None


def test_verdict_confidences_are_probabilities(parser: Parser) -> None:
    release = parser.parse("Show.S01E03.1080p.WEB-DL.x264-GRP.mkv")
    for value in (release.content_confidence, release.numbering_confidence,
                  release.pack_scope_confidence, release.special_confidence,
                  release.adult_confidence):
        assert 0.0 < value <= 1.0
    # A clean scene episode is not a borderline call.
    assert release.numbering_confidence > 0.9


def test_languages_and_subtitles(parser: Parser) -> None:
    release = parser.parse("Die.Discounter.S04E03.German.DL.1080p.WEB.h265-GRP")
    assert release.language == ("deu",)
    dual = parser.parse("Perfect Days 2023 720p BluRay x264 DuaL-TURKO")
    assert any(origin.field is OriginField.DUAL_AUDIO for origin in dual.origins)
    subs = parser.parse("[Judas] Show - S01E01 [1080p][HEVC x265][Multi-Subs].mkv")
    assert subs.subtitle_format == SubtitleFormat.UNKNOWN  # format only when the name states one


def test_origins_carry_text_ranges_and_confidence(parser: Parser) -> None:
    name = "Show.S01E03.1080p.WEB-DL.x264-GRP.mkv"
    release = parser.parse(name)
    assert release.origins, "a parsed name must carry evidence"
    raw = name.encode("utf-8")
    for origin in release.origins:
        assert isinstance(origin, Origin)
        if origin.text:  # located spans; whole-name verdicts carry no text
            assert 0 <= origin.begin < origin.end <= len(raw)
            assert raw[origin.begin:origin.end].decode("utf-8") == origin.text
        assert 0.0 <= origin.confidence <= 1.0


def test_unicode_round_trips_unmangled(parser: Parser) -> None:
    release = parser.parse("[桜都字幕组] 葬送的芙莉莲 / Sousou no Frieren [28][1080p]")
    assert release.title == "Sousou no Frieren"
    assert "桜都字幕组" in release.release_group
    cyrillic = parser.parse("Эпидемия.S02E05.2022.WEB-DL.1080p.RUS.LostFilm")
    assert cyrillic.title == "Эпидемия"
    assert cyrillic.language == ("rus",)


# --- hostile input through the binding -------------------------------------------------------------

def test_hostile_inputs_return_invalid_not_raise(parser: Parser) -> None:
    for name in ("", " ", "\t", "...", "()[]", "\x00abc", "🎬🎬"):
        release = parser.parse(name)
        assert isinstance(release, ParsedRelease)
        assert not release.valid, repr(name)
        assert release.title is None, repr(name)


def test_malformed_utf8_is_replaced_not_fatal(parser: Parser) -> None:
    # A lone continuation byte cannot be produced by str; go through a surrogate escape.
    mangled = b"Movie\x80.2024.1080p.WEB-DL.x264-GRP".decode("utf-8", "surrogateescape")
    release = parser.parse(mangled)
    assert release.year == 2024
    batch = parser.parse_batch([mangled])
    assert batch[0].year == 2024


def test_batch_matches_single_and_keeps_order(parser: Parser) -> None:
    names = ["Show.S01E01.1080p.WEB-DL.x264-A.mkv",
             "Movie.2024.2160p.BluRay.x265-B.mkv",
             "Show.S01E02.1080p.WEB-DL.x264-A.mkv"]
    batch = parser.parse_batch(names)
    assert [entry.release_group for entry in batch] == [("A",), ("B",), ("A",)]
    assert batch[1] == parser.parse(names[1])  # dataclass equality, every field
    assert parser.parse_batch([]) == []


# --- construction and errors -----------------------------------------------------------------------

def test_origins_default_on_and_can_be_skipped(parser: Parser) -> None:
    name = "Show.S01E03.1080p.WEB-DL.x264-GRP.mkv"
    full = parser.parse(name)
    assert full.origins
    assert parser.parse_batch([name])[0] == full        # same default both paths
    slim = parser.parse(name, origins=False)
    assert slim.origins == ()
    from dataclasses import replace
    # Only the evidence differs - and the two fields read from it.
    assert replace(full, origins=(), screen_size_text=None, frame_size=None) == slim   # values compare as values
    assert slim.title.confidence == 1.0 and full.title.confidence <= 1.0
    assert slim.episode_title == full.episode_title   # title facts are view fields, not evidence
    assert slim.screen_size == full.screen_size         # the tier does not depend on the evidence


def test_default_construction_finds_the_models() -> None:
    with Parser() as parser:
        assert parser.parse("Movie.2024.1080p.WEB-DL.x264-GRP.mkv").valid


def test_threads_option_is_accepted() -> None:
    with Parser(threads=2) as parser:
        assert len(parser.parse_batch(["A.2024.mkv", "B.2024.mkv"])) == 2


def test_missing_models_directory_raises_with_detail(tmp_path: Path) -> None:
    with pytest.raises(NeureleaseError):
        Parser(tmp_path)  # exists, but holds no model files


def test_model_directory_with_non_ascii_characters(tmp_path: Path) -> None:
    # Regression: the segmenter path round-tripped through path.string(), which re-encodes to the
    # ANSI codepage on Windows, so a directory with an umlaut in it failed to open exactly one of
    # the four model files.
    import shutil
    target = tmp_path / "modelle_übung_日本"
    source = Path(__file__).resolve().parents[3] / "model"
    shutil.copytree(source, target)
    with Parser(target) as parser:
        assert parser.parse("Movie.2024.1080p.WEB-DL.x264-GRP.mkv").valid


def test_byte_cap_boundary(parser: Parser) -> None:
    base = "Movie.2024.1080p."
    at_cap = base + "a" * (16 * 1024 - len(base))
    within = parser.parse(at_cap)
    assert within.valid and not within.degraded
    # One byte over: the model is skipped and the title/year heuristics answer instead --
    # degraded and visible, never silent, never a crash.
    over = parser.parse(at_cap + "a")
    assert over.degraded
    assert over.year == 2024
    multibyte = "ü" * (8 * 1024)                   # exactly 16 KB of UTF-8
    assert not parser.parse(multibyte).degraded
    assert parser.parse(multibyte + "x").degraded


def test_determinism(parser: Parser) -> None:
    name = "Chaos.Walking.2021.2160p.UHD.BluRay.REMUX.DV.TrueHD.Atmos.H264-BTM"
    reference = parser.parse(name)
    for _ in range(50):
        assert parser.parse(name) == reference


def test_parsers_are_independent_across_threads() -> None:
    import threading
    failures: list[BaseException] = []

    def work() -> None:
        try:
            with Parser() as parser:
                for _ in range(100):
                    assert parser.parse("Show.S01E03.1080p.WEB-DL.x264-GRP.mkv").season == 1
        except BaseException as error:  # noqa: BLE001
            failures.append(error)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert not failures, failures[0]


def test_bogus_library_path_raises() -> None:
    with pytest.raises(NeureleaseError):
        Parser(library=os.devnull)


def test_closed_values_are_their_labels(parser: Parser) -> None:
    """A Kind member is the enum AND its label: identity, string equality, str(), JSON."""
    import json
    release = parser.parse("Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-TERMiNAL.mkv")
    assert release.source == SourceKind.BLURAY and release.source.kind == SourceKind.BLURAY
    assert release.source == "BluRay" and str(release.source) == "BluRay" and f"{release.source}" == "BluRay"
    assert release.source.name == "BLURAY" and release.source.id == 1
    assert release.screen_size == "2160p" and release.video_codec == "HEVC"
    assert json.dumps({"source": release.source}) == '{"source": "BluRay"}'
    assert SourceKind.from_id(1) is SourceKind.BLURAY


def test_screen_size_text_and_frame_size_come_from_the_evidence(parser: Parser) -> None:
    exact = parser.parse("[Doki] Onii-chan Dakedo Ai Sae Areba Kankeinai yo ne - Vol 2 (1920x1080 Hi10P BD FLAC)")
    assert exact.screen_size == ResolutionTier.P1080 and exact.screen_size.kind == ResolutionTier.P1080
    assert exact.screen_size_text == "1920x1080" and exact.frame_size == (1920, 1080)
    tier = parser.parse("Gladiator.EXTENDED.2000.720.BrRip.264.YIFY")
    assert tier.screen_size == ResolutionTier.P720 and tier.screen_size.kind == ResolutionTier.P720 and tier.screen_size_text == "720" and tier.frame_size is None
    assert parser.parse("Gladiator.EXTENDED.2000.720.BrRip.264.YIFY", origins=False).screen_size_text is None


def test_to_dict_uses_guessit_property_names_for_shared_facts(parser: Parser) -> None:
    import json
    release = parser.parse("Show.S00E01.Special.1080p.WEB-DL.x264-GRP.mkv")
    plain = release.to_dict()
    assert plain["title"] == "Show" and plain["season"] == 0 and plain["episode"] == 1
    assert plain["screen_size"] == "1080p" and plain["source"] == "WEB-DL" and plain["video_codec"] == "H264"
    assert plain["release_group"] == "GRP" and plain["container"] == "mkv"       # one value: a scalar
    assert plain["type"] == "episode" and plain["content"] == "series"
    assert plain["adult"] is False and plain["special"] == "special"
    assert "year" not in plain and "hdr" not in plain and "origins" not in plain
    assert set(plain["confidence"]) >= {"title", "season", "episode", "screen_size", "source", "content"}
    json.dumps(plain)
    ranged = parser.parse("HI.SCORE.GIRL.II.S01E01-E09.Blu-ray.1080p.x264").to_dict()
    assert ranged["episode"] == list(range(1, 10))                                # a range: the list
    audio = parser.parse("Movie.2024.2160p.UHD.AMZN.WEB-DL.HEVC.DDP5.1.Atmos.DV-GRP.mkv").to_dict()
    assert audio["audio_codec"] == "DDP" and audio["audio_channels"] == "5.1" and audio["audio_profile"] == "Atmos"
    assert audio["streaming_service"] == "AMZN" and "Dolby Vision" in (audio["other"] if isinstance(audio["other"], list) else [audio["other"]])
    with_evidence = release.to_dict(origins=True)
    assert with_evidence["origins"][0]["field"] == "title" and with_evidence["origins"][0]["text"] == "Show"


def test_title_facts_from_the_evidence(parser: Parser) -> None:
    show = parser.parse("Power.Rangers.Megaforce.S01E04.Der.falsche.Ranger.German.DL.720p.BluRay.x264-TV4A")
    assert show.episode_title == "Der falsche Ranger"
    anime = parser.parse("[SubsPlease] Shingeki no Kyojin (Attack on Titan) - 87 (1080p) [A1B2C3D4].mkv")
    assert anime.title == "Attack on Titan" and anime.alternative_title == ("Shingeki no Kyojin",)
    assert anime.absolute_episode == 87 and anime.episode is None      # absolute, not S?E0
    assert show.to_dict()["episode_title"] == "Der falsche Ranger"


@pytest.mark.xfail(strict=False, reason="statement of intent: the 2026-09-07 weights read a "
                   "franchise prefix and its title as one span on 92 of 92 labelled names, so "
                   "`James Bond 007` comes back inside the title rather than beside it")
def test_franchise_prefix_is_separated_from_the_title(parser: Parser) -> None:
    bond = parser.parse("James.Bond.007.Casino.Royale.2006.German.DL.1080p.BluRay.x264-DETAiLS")
    assert bond.title == "Casino Royale" and bond.franchise_prefix == "James Bond 007"
    assert bond.to_dict()["franchise_prefix"] == "James Bond 007"


def test_every_value_carries_its_confidence(parser: Parser) -> None:
    release = parser.parse("Gladiator.EXTENDED.2000.720.BrRip.264.YIFY")
    assert release.title == "Gladiator" and 0.0 < release.title.confidence <= 1.0
    assert release.year == 2000 and release.year + 1 == 2001 and release.year.confidence > 0.9
    assert release.screen_size == "720p" and release.screen_size.kind == ResolutionTier.P720
    assert release.screen_size.confidence < 1.0                    # the bare 720 is the least sure span
    assert release.release_group == ("YIFY",) and release.release_group[0].confidence > 0.9
    assert release.release_group.confidence == min(g.confidence for g in release.release_group)
    assert release.content.confidence == release.content_confidence
    plain = release.to_dict()
    assert plain["confidence"]["title"] == round(release.title.confidence, 3)
    assert plain["confidence"]["screen_size"] == round(release.screen_size.confidence, 3)
    assert "year" in plain["confidence"] and "season" not in plain["confidence"]
