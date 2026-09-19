# NeuRelease

**A fast neural parser for torrent and release names.**

[![build](https://github.com/bonejay/neurelease/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/bonejay/neurelease/actions/workflows/build.yml)
![platforms](https://img.shields.io/badge/Linux%20%7C%20Windows%20%7C%20macOS-555)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![demo](https://img.shields.io/badge/demo-try%20it%20in%20the%20browser-2f62f0)](https://neurelease-demo.vercel.app)

NeuRelease is a fast, high-performance, multilingual neural parser for torrent and release names.
It reads a name, extracts its fields - title, season and episode, year, quality, codecs, languages,
release group and more - and classifies what the name is: a film, a series, music, a game,
software, a book, and whether it is anime. Every value comes with a confidence score and the span
it was read from.

It combines a character-level CNN with a Transformer encoder, running as int8 inference with
runtime-dispatched SIMD kernels, behind C++, C and Python APIs. No ML runtime; nothing to download.

[![What the model read: every field of an anime release name, with its span and confidence](docs/images/what-the-model-read.png)](https://neurelease-demo.vercel.app)

Fifteen fields read out of one name, each shown against the characters it came from.
[Try it in the browser](https://neurelease-demo.vercel.app).

Pattern-based parsers recognise known markers and guess the rest by position, so anything
ambiguous - a number that may be a year or an episode, a word that may be a language or part of
the title - is settled the same way every time, right or wrong. NeuRelease decides from context. It
is trained on hundreds of thousands of real, labelled release names from a large torrent index -
mostly English, with German, Spanish, French, Italian, Russian, Chinese and Japanese names as well.
It is about 3.4× faster than GuessIt on one thread, ~10× in batch, and should come out ahead on most
real-world names.

## Use

```python
from neurelease import Parser

parser = Parser()
r = parser.parse("Ted.Lasso.S03E03.4-5-1.1080p.ATVP.WEB-DL.DDP5.1.H.264-NTb")

r.title                                # 'Ted Lasso'
r.season, r.episode, r.episode_title   # 3, 3, '4-5-1'
r.streaming_service, r.release_group   # 'ATVP', ('NTb',)
r.source == "WEB-DL"                   # True: enums equal their labels
r.year                                 # None: the name does not say
r.title.confidence, r.episode_title.confidence  # 1.00, 0.80: every value knows how sure the model was
r.to_dict()                            # {'title': 'Ted Lasso', 'season': 3, 'episode': 3, ...}

names = ["Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-TERMiNAL.mkv",
         "Stray_v1.5-Razor1911",
         "La Casa de Papel - Temporada 3 [HDTV 720p][Cap.305][AC3 5.1 Castellano]",
         "【高清剧集网 www.BTHDTV.com】邻家哥哥给我爱[第05-06集][简繁英字幕].Brother.Next.Door.2024.S01E05-06.1080p.WEB-DL.H264.AAC-BTHDTV",
         "葬送のフリーレン 第28話 「また会ったときに恥ずかしいからね」 (1080p).mkv",
         "Blade.Runner.1982.Final.Cut.REPACK.2160p.UHD.BluRay.DV.HDR.TrueHD.7.1-FraMeSToR"]
releases = parser.parse_batch(names)   # one result per name, in order

# Editions, revisions and the flags a name states about ITSELF rather than its content.
r = parser.parse("Dune.Part.Two.2024.PROPER.REPACK.2160p.UHD.BluRay.REMUX.DV.HDR.TrueHD.7.1-FraMeSToR")
r.proper, r.repack                     # True, True
r.release_version                      # 2: a proper or a repack IS the second version
r.remux, r.hdr                         # True, 'DV'
r.to_dict()["other"]                   # ['Proper', 'Repack', 'Remux', 'HDR10', 'Dolby Vision']

r = parser.parse("The.Wire.S01E01.INTERNAL.RESTORED.1080p.BluRay.x265-SARTRE")
r.edition                              # ('Internal', 'Restored'): a name may state several
r = parser.parse("Nosferatu.1922.RESTORED.Colorized.1080p.BluRay.x264-CiNEFiLE")
r.edition                              # ('Colorized', 'Restored'): a restoration is not a remaster
r = parser.parse("Knight.Rider.2000.1991.OAR.GERMAN.DL.BDRIP.X264-WATCHABLE")
r.edition                              # ('Original Aspect Ratio',): not reframed, whatever shape

SHOW = ("title", "alternative_title", "season", "episode", "absolute_episode",
        "episode_title", "year", "content", "edition", "version")
for r in releases:
    d = r.to_dict()
    print({k: d[k] for k in SHOW if d.get(k) is not None})
# {'title': 'Blade Runner 2049', 'year': 2017, 'content': 'movie'}
# {'title': 'Stray', 'content': 'game'}
# {'title': 'La Casa de Papel', 'season': 3, 'episode': 5, 'content': 'series'}
# {'title': 'Brother Next Door', 'alternative_title': '邻家哥哥给我爱', 'season': 1, 'episode': [5, 6], 'year': 2024, 'content': 'series'}
# {'title': '葬送のフリーレン', 'absolute_episode': 28, 'episode_title': 'また会ったときに恥ずかしいからね', 'content': 'series'}
# {'title': 'Blade Runner', 'year': 1982, 'content': 'movie', 'edition': 'Final Cut', 'version': 2}
```

```sh
pip install neurelease
```

[The package is on PyPI](https://pypi.org/project/neurelease/); the wheel bundles the built library
and the model files, so `Parser()` needs no paths, downloads nothing and works from any directory.
To build it yourself instead, `pip install ./bindings/python` after building the library (see
Build). `parse_batch` runs many names at once on several threads and returns them in input order.
The C++ and C APIs: [docs/API.md](docs/API.md).
Everything a result contains, field by field: [docs/RESULT.md](docs/RESULT.md). Titles and evidence
keep the original script - Latin, Han, Kana, Cyrillic.

## Compared with GuessIt, Sonarr and Radarr

Measured on 2026-09-19 with the shipped model (version 4). Every figure is the share of names
answered COMPLETELY correctly - every field the name states read right, nothing invented. One
wrong field fails the name.

Two things make a four-parser comparison unfair unless they are removed, and they compound.
**Domain**: Sonarr answers series and Radarr answers film, and each returns nothing at all for the
other, so a mixed set charges both for names they never claimed. **Vocabulary**: Sonarr models 16
fields and Radarr 14, against 29 here and 28 in GuessIt, so a case asserting a codec or a container
fails them by construction. So the table below is split by content kind and scored only on the nine
fields all four answer: `work_title`, `year`, `resolution`, `source_family`, `release_group`,
`release_variant`, `audio_language`, `subtitle_language`, `crc32`.

| | cases | NeuRelease | GuessIt | Radarr | Sonarr |
|---|---:|---:|---:|---:|---:|
| **our labelled names** | | | | | |
| representative, films | 1,001 | **93.5%** | 78.0% | 69.7% | 2.5% |
| representative, series | 2,075 | **96.1%** | 60.9% | 8.1% | 65.5% |
| hard, films | 1,736 | **63.3%** | 36.7% | 45.7% | 1.2% |
| hard, series | 1,794 | **79.3%** | 41.0% | 4.3% | 48.7% |
| **each parser's own suite** | | | | | |
| GuessIt's corpus, films | 194 | 88.1% | **95.4%** | 49.5% | 2.1% |
| GuessIt's corpus, series | 461 | 87.2% | **94.4%** | 7.2% | 57.3% |
| Sonarr's suite, series | 935 | 89.5% | 80.7% | 45.0% | **95.2%** |
| Radarr's suite, films | 535 | 85.8% | 76.8% | **98.5%** | 68.8% |

Read the two halves differently. The first is our own gold - we chose the names, wrote the labels
and fixed the contract, and NeuRelease is developed against them, so a lead there is expected. The
second belongs to the other parsers: written to pin down their own behaviour, and nobody here
trained on them. Each parser wins its own suite; NeuRelease is second on all three and first on
none, which is what a parser written against none of them should look like.

On the full twenty-field contract, video names, our validation split: **97.86% macro field F1 and
90.04% exact** against GuessIt's 86.67% and 52.72%. It is about **3.3x faster on one thread**
(2,604 us/name against 8,621) and reaches 711 us/name in batch, which GuessIt has no API for.
Of GuessIt's 22 documented limitation cases it solves **19**; GuessIt solves 0, Sonarr 7.

**What is filtered, and why.** GuessIt's corpus is its own test suite - fixture strings such as
`FooBar.307.PDTV-FlexGet`, written to exercise its rules - so three kinds of entry are left out
rather than scored: those written as filesystem PATHS, whose expectations come from parent
directories a release name does not carry; those asserting `type`, which every entry inherits from
its file's defaults and which assumes the input is video before anything has read it; and checks
naming a value our closed vocabulary cannot express. 859 of 1,048 entries are scored, and what is
dropped is listed in the full report rather than quietly excluded. Sonarr's and Radarr's suites are
scored almost unfiltered: their expectations are stated per case rather than inherited, so only
assertions about things this vocabulary has no field for are left out.

Name-by-name comparisons, where the difference is visible rather than averaged:
[docs/ANIME.md](docs/ANIME.md) and [docs/LIVE_ACTION.md](docs/LIVE_ACTION.md).

Method, exact model identity, scoring snapshot and reproduction:
[docs/GUESSIT_COMPARISON.md](docs/GUESSIT_COMPARISON.md).
Hardware and native kernel timings: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build

Requirements: CMake 3.24+, a C++23 compiler, and Ninja or another CMake generator. PCRE2 is the
only third-party library dependency and is fetched automatically when not installed.

```sh
git clone https://github.com/bonejay/neurelease.git
cd neurelease
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
python -m pytest bindings/python/tests
```

`cmake --install build/release --prefix dist` produces a self-contained package. Build options,
the optional GuessIt-corpus test, and benchmark instructions are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

| | |
|---|---|
| [docs/API.md](docs/API.md) | Python, C++ and C usage |
| [docs/RESULT.md](docs/RESULT.md) | Every result field and its conventions |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Model, conversion layer, performance, build options, model versions |
| [docs/C_ABI.md](docs/C_ABI.md) | The C binary interface |
| [docs/GUESSIT_COMPARISON.md](docs/GUESSIT_COMPARISON.md) | Method and per-field numbers of the GuessIt comparison |
| [docs/ANIME.md](docs/ANIME.md) | The anime verdict, and five anime names read by both parsers |
| [docs/LIVE_ACTION.md](docs/LIVE_ACTION.md) | Fourteen live-action names, in four languages, read by both parsers |

## License

[MIT](LICENSE).

PCRE2 is statically linked into the library and travels inside every binary this project
distributes, including the Python wheels; its licence is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
