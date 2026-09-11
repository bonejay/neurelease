from pathlib import Path
import os

from neurelease import ContentKind, OriginField, Parser, ResolutionTier, SourceKind


def test_single_and_batch() -> None:
    # Env overrides when set; otherwise the binding's own discovery is part of what is tested.
    library = os.environ.get("NEURELEASE_LIBRARY")
    models = os.environ.get("NEURELEASE_MODELS")
    with Parser(Path(models) if models else None, library) as parser:
        movie = parser.parse("Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-GRP.mkv")
        assert movie.valid
        assert movie.screen_size == ResolutionTier.P2160
        assert movie.source == SourceKind.BLURAY
        assert movie.content == ContentKind.MOVIE
        # The medium is no longer guessed at; the tradition is, and a Blade Runner rip is not it.
        assert movie.anime is False
        assert movie.release_group == ("GRP",)
        assert movie.container == "mkv"
        assert not movie.specials
        assert movie.season_end is None      # a movie states no season range
        assert movie.absolute_episode is None
        assert movie.episode_count is None
        title = next(origin for origin in movie.origins if origin.field is OriginField.TITLE)
        assert title.text == "Blade.Runner.2049"

        results = parser.parse_batch([
            "Game.of.Thrones.S01E08.1080p.WEB-DL.x264-GRP.mkv",
            "Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-GRP.mkv",
        ])
        assert len(results) == 2
        assert results[0].content == ContentKind.SERIES
        assert results[1].content == ContentKind.MOVIE
