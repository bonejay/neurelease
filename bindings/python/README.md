# NeuRelease

**A fast neural parser for torrent and release names.**

[![build](https://github.com/bonejay/neurelease/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/bonejay/neurelease/actions/workflows/build.yml)
![platforms](https://img.shields.io/badge/Linux%20%7C%20Windows%20%7C%20macOS-555)
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/bonejay/neurelease/blob/main/LICENSE)
[![demo](https://img.shields.io/badge/demo-try%20it%20in%20the%20browser-2f62f0)](https://neurelease-demo.vercel.app)

NeuRelease is a fast, high-performance, multilingual neural parser for torrent and release names.
It reads a name, extracts its fields - title, season and episode, year, quality, codecs, languages,
release group and more - and classifies what the name is: a film, a series, music, a game,
software, a book, and whether it is anime. Every value comes with a confidence score and the span
it was read from.

It combines a character-level CNN with a Transformer encoder, running as int8 inference with
runtime-dispatched SIMD kernels. No ML runtime, no model download: the wheel carries the compiled
library and the weights, about 5 MB, and `Parser()` needs no paths.

Pattern-based parsers recognise known markers and guess the rest by position, so anything
ambiguous - a number that may be a year or an episode, a word that may be a language or part of
the title - is settled the same way every time, right or wrong. NeuRelease decides from context. It
is trained on hundreds of thousands of real, labelled release names from a large torrent index -
mostly English, with German, Spanish, French, Italian, Russian, Chinese and Japanese names as well.

```sh
pip install neurelease
```

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

`parse_batch` runs many names at once on several threads and returns them in input order. Titles
and evidence keep the original script - Latin, Han, Kana, Cyrillic.

A name also states things about ITSELF rather than its content, and those are fields too:

```python
r = parser.parse("The.Expanse.S05E06.REAL.PROPER.1080p.AMZN.WEB-DL.DDP5.1.H.264-NTb")
r.proper, r.release_real, r.release_version   # True, 1, 2
```

`release_version` is the one number to sort on: a proper or a repack IS the second copy, `v2` is
the anime spelling of the same thing, and `release_real` counts the times a botched proper had to
be redone - the scene writes `REAL.REAL.PROPER`, and a boolean would read that as the same release.
`edition` is a tuple because a release is routinely several at once. Every field a result can carry
is documented in
[docs/RESULT.md](https://github.com/bonejay/neurelease/blob/main/docs/RESULT.md).

## Compared with GuessIt, Sonarr and Radarr

Measured on 2026-09-19 with the shipped model (version 4). Every figure is the share of names
answered COMPLETELY correctly - every field the name states read right, nothing invented. One
wrong field fails the name.

Sonarr answers series and Radarr answers film, and each returns nothing at all for the other, so
the table is split by content kind. Sonarr also models 16 fields and Radarr 14, against 29 here, so
it is scored only on the nine all four answer: title, year, resolution, source, release group,
edition, audio language, subtitle language, checksum.

| | cases | NeuRelease | GuessIt | Radarr | Sonarr |
|---|---:|---:|---:|---:|---:|
| our labelled names, films | 1,001 | **93.5%** | 78.0% | 69.7% | 2.5% |
| our labelled names, series | 2,075 | **96.1%** | 60.9% | 8.1% | 65.5% |
| our hard names, films | 1,736 | **63.3%** | 36.7% | 45.7% | 1.2% |
| our hard names, series | 1,794 | **79.3%** | 41.0% | 4.3% | 48.7% |
| GuessIt's own corpus, films | 194 | 88.1% | **95.4%** | 49.5% | 2.1% |
| GuessIt's own corpus, series | 461 | 87.2% | **94.4%** | 7.2% | 57.3% |
| Sonarr's own suite, series | 935 | 89.5% | 80.7% | 45.0% | **95.2%** |
| Radarr's own suite, films | 535 | 85.8% | 76.8% | **98.5%** | 68.8% |

Read the halves differently. The first four rows are our own labelled names - we chose them, wrote
the labels and fixed the contract, and NeuRelease is developed against them, so a lead there is
expected. The last four belong to the other parsers, written to pin down their own behaviour, and
nobody here trained on them. Each parser wins its own suite; NeuRelease is second on all three and
first on none, which is what a parser written against none of them should look like.

On the full twenty-field contract, video names: **97.86% macro field F1 and 90.04% exact** against
GuessIt's 86.67% and 52.72%. About **3.3x faster on one thread** (2,604 us/name against 8,621) and
711 us/name in batch, which GuessIt has no API for. Of GuessIt's 22 documented limitation cases it
solves **19**; GuessIt solves 0.

GuessIt's corpus is its own test suite - fixture strings such as `FooBar.307.PDTV-FlexGet`, written
to exercise its rules - so entries written as filesystem paths are left out, along with `type`
assertions inherited from each file's defaults, which assume the input is video before anything has
read it. 859 of 1,048 entries are scored. Sonarr's and Radarr's suites are scored almost
unfiltered: their expectations are stated per case rather than inherited.

Method, exact model identity, scoring snapshot and reproduction:
[docs/GUESSIT_COMPARISON.md](https://github.com/bonejay/neurelease/blob/main/docs/GUESSIT_COMPARISON.md).

## Wheels

Python 3.10 or newer. The wheels are tagged `py3-none-<platform>`, one per platform rather than one
per interpreter, because the library is loaded through `ctypes` and speaks the stable C ABI.

| Platform | Wheel |
|---|---|
| Linux x86-64 | `manylinux_2_28` |
| macOS, Apple Silicon | `macosx_11_0_arm64` |
| Windows x86-64 | `win_amd64` |

Anywhere else, build from source: the C++ library, the C ABI and the build instructions are in the
[repository](https://github.com/bonejay/neurelease).

## License

[MIT](https://github.com/bonejay/neurelease/blob/main/LICENSE). PCRE2 is statically linked into the
bundled library, and its notice travels in the wheel as `THIRD_PARTY_NOTICES.md`.
