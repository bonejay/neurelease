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
         "El Joven Sheldon - Temporada 6 [HDTV 720p][Cap.604][AC3 5.1 Castellano][www.pctnew.org]",
         "【高清剧集网 www.BTHDTV.com】邻家哥哥给我爱[第05-06集][简繁英字幕].Brother.Next.Door.2024.S01E05-06.1080p.WEB-DL.H264.AAC-BTHDTV",
         "葬送のフリーレン 第28話 「また会ったときに恥ずかしいからね」 (1080p).mkv"]
releases = parser.parse_batch(names)   # one result per name, in order

SHOW = ("title", "alternative_title", "season", "episode", "absolute_episode", "episode_title", "year", "content")
for r in releases:
    d = r.to_dict()
    print({k: d[k] for k in SHOW if d.get(k) is not None})
# {'title': 'Blade Runner 2049', 'year': 2017, 'content': 'movie'}
# {'title': 'Stray', 'content': 'game'}
# {'title': 'El Joven Sheldon', 'season': 6, 'episode': 4, 'content': 'series'}
# {'title': 'Brother Next Door', 'alternative_title': '邻家哥哥给我爱', 'season': 1, 'episode': [5, 6], 'year': 2024, 'content': 'series'}
# {'title': '葬送のフリーレン', 'absolute_episode': 28, 'episode_title': 'また会ったときに恥ずかしいからね', 'content': 'series', 'anime': True}
```

`parse_batch` runs many names at once on several threads and returns them in input order. Titles
and evidence keep the original script - Latin, Han, Kana, Cyrillic. Every field a result can carry
is documented in
[docs/RESULT.md](https://github.com/bonejay/neurelease/blob/main/docs/RESULT.md).

## Compared with GuessIt

Measured on the same machine on 2026-09-11 with the shipped model (version 3), on 3,344 video
validation names:

| Metric, video only (3,344 names, 2026-09-11) | NeuRelease | GuessIt 4.4.0 |
|---|---:|---:|
| macro shared-field F1 | **97.57%** | 86.45% |
| exact on every applicable shared field | **89.44%** | 51.44% |
| single name, one thread, via Python | **2,446 us/name** | 8,213 us/name |
| batch, 4 workers, via Python | **833 us/name** | no batch API |
| GuessIt's 22 documented limitation cases solved | **19/22** | 0/22 |
| GuessIt's own published regression corpus | **683/859** | 804/859 |

About **3.4× faster on one thread** in this measurement, and roughly ten times in batch.

GuessIt's regression corpus is its own test suite: fixture strings such as `FooBar.307.PDTV-FlexGet`
and filesystem paths, written to exercise its rules, with every input assumed to be a video.
NeuRelease parses a single release name as found in real traffic and classifies it before assuming
anything, so on this corpus it scores 683 to 804 - and on real names the ranking reverses. The 22
limitation cases are GuessIt's own documented failures, not a representative sample.

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
