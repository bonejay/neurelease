"""Ranges across seasons, against names labelled by hand.

tests/data/cross_season_ranges.jsonl holds every `SxxEyy-SxxEzz` and `NxNN-NxNN` name spanning
several seasons found in the magnetdb, predb and Anime Tosho corpora, each labelled with whether
its episode numbers run on across the seasons or restart in each, and the evidence: the torrent's
file count, the text itself, or what is known of the show.

The parser may withhold episode numbers it cannot be sure of - the seasons still say what the pack
is - but it must never answer episodes for a pack that numbers them per season.
"""

import json
from pathlib import Path

import pytest

from neurelease import Parser

CASES = [json.loads(line) for line in
         (Path(__file__).resolve().parents[3] / "tests/data/cross_season_ranges.jsonl")
         .read_text(encoding="utf-8").splitlines() if line.strip()]
WITHHELD_FROM_RUNNING = 4


@pytest.fixture(scope="module")
def parser():
    with Parser() as instance:
        yield instance


def number(value):
    return None if value is None else int(value)


@pytest.mark.parametrize("case", CASES, ids=[case["name"][:60] for case in CASES])
def test_seasons_always_and_episodes_never_wrong(parser: Parser, case: dict) -> None:
    result = parser.parse(case["name"])
    assert (number(result.season), number(result.season_end)) == (case["season"], case["season_end"])
    if result.episode is not None:
        assert case["numbering"] != "per_season", "episodes answered for a pack numbered per season"
        if case["numbering"] == "running":
            assert (number(result.episode), number(result.episode_end)) == (case["episode"],
                                                                            case["episode_end"])


def test_how_many_running_packs_go_unanswered(parser: Parser) -> None:
    withheld = [case["name"] for case in CASES
                if case["numbering"] == "running" and parser.parse(case["name"]).episode is None]
    assert len(withheld) == WITHHELD_FROM_RUNNING, withheld
