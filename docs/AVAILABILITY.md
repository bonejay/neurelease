# Availability

`availability` is one number from 0 to 1 per release name: **how likely this release is to be easy
to get**. A release is easy to get when it is already in a debrid cache, or when enough people keep
it that a cache would be likely to hold it.

```python
from neurelease import Parser

parser = Parser()
r = parser.parse("Paradise.2025.S01.1080p.10bit.WEBRip.6CH.x265.HEVC-PSA")
r.availability                 # 0.89
r.to_dict()["availability"]    # 0.89, beside the other facts
```

```cpp
const neurelease::ParseResult result = parser.parse(name);
result.info.availability;      // 0.89F; -1 on a model without the head
```

```c
float a = rp_availability(result);   /* 0.89f; -1 on a model without the head */
/* also rp_result_view.availability, for bindings that read everything in one call */
```

## What goes in

**The name, and nothing else.** `parse()` takes one string. Seeders, leechers, size, upload date and
the title's popularity are not inputs and cannot be passed. That is on purpose. Index seeder counts
are often stale or made up, and a caller who has better facts can combine them with this number.

Everything the number knows comes from how the name is written: the group, the source, the
resolution and codec, whether it is a season pack or a single episode, the tracker or site tags,
and how carefully the name follows its scene's conventions.

## Reading the number

Six names for two titles, as the shipped model reads them:

| availability | release |
|---:|---|
| 0.98 | `Love.Through.a.Prism.S01.1080p.NF.WEB-DL.MULTi.Opus5.1.AV1.HDR10P-Trix` |
| 0.98 | `Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX.DV.HDR.HEVC.TrueHD.7.1.Atmos-FGT` |
| 0.95 | `Dune Part Two (2024) [2160p] [4K] [WEB] [5.1] [YTS.MX]` |
| 0.19 | `Love.Through.a.Prism.S01E01.720p.HEVC.x265-MeGusta` |
| 0.14 | `Love Through a Prism S01E01 XviD-AFG` |
| 0.04 | `[AniDub]_Love.Through.a.Prism.s01` |

A complete season from a known group, written in the usual way, scores high. So do a popular
remux and a popular small encode. A single old episode, an XviD rip and an oddly written dub pack
score low.

On names it never trained on, the number reads roughly as a probability: of all releases near
0.9, about 85 % were really cached, and of those near 0.05, about 5 % (table below). It does not
reach full certainty at either end. A release at 0.5 is cached about half the time.

## Using it

**To rank.** Use it to order candidates when nothing better is known, or to break ties after your
own rules:

```python
candidates = [parser.parse(name) for name in names]
best_first = sorted(candidates, key=lambda r: r.availability or 0.0, reverse=True)
```

**To decide what to ask a cache about first.** Some debrid services can only be probed one
torrent at a time. Ask about the likeliest first.

**Together with your own facts.** If you know the size, the running time or a cache's real
answer, combine them with this number in your ranking. A real cache answer beats this estimate
every time. Size per hour of running time is the natural counterweight to small encodes (see
Known limits).

**Never as "not asked".** `None` in Python and `-1` in C/C++ mean that the installed model has no
availability head. That says nothing about the release. Model 6 (2026-11-17) is the first model
with the head.

## What it was learned from

The model has a small extra head trained together with the parser, on the same names. It has its
own summary token in the transformer and its own small output stack, and it shares the rest of the
model. Its training target was one fact per name: did TorBox, a debrid service, hold that exact
torrent in its cache when asked? 176,763 real torrents were checked in November 2026, and 28 %
were cached. Names with no real torrent behind them took no part in training. A name therefore
counts as unavailable only when its torrent was asked about and was not held.

A cache holds what people ask for and what stays seeded. The head therefore learns what wanted,
well-kept releases look like. That is why the field is called availability and not cache
probability: it is useful beyond one service. It is also not a quality judgement. `screen_size`,
`source` and `video_codec` describe the picture, and `availability` describes how easy the release
is to get.

The parser did not pay for the head. Next to the same training without it, every parser measure is
within 0.005, the int8 runtime scores the same, and GuessIt's corpus reads 672 cases fully correct
against 670.

## How well it works

All of these numbers come from names the model never trained on.

**Validation names.** 11,871 real torrents whose whole title family was kept out of training,
23 % cached, scored through the shipped int8 C++ runtime:

| | |
|---|---:|
| AUC (the chance a cached name scores above an uncached one) | 0.883 |
| cached among the top tenth by availability | 81 % |
| cached among the bottom tenth | 0.3 % |
| Brier score (mean squared error of the probability) | 0.110 |

How often a release really was cached, by its availability:

| availability | 0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| names | 6,481 | 1,208 | 662 | 434 | 523 | 436 | 429 | 533 | 559 | 606 |
| really cached | 5 % | 17 % | 23 % | 32 % | 42 % | 46 % | 52 % | 66 % | 78 % | 85 % |

It does more than recognise popular-looking names. The AUC holds within each seeder band, which
was measured but never shown to the model:

| seeders | names | cached | AUC |
|---|---:|---:|---:|
| 50+ | 150 | 75 % | 0.934 |
| 5-49 | 3,398 | 39 % | 0.888 |
| 1-4 | 6,928 | 16 % | 0.884 |
| no swarm data (Anime Tosho) | 1,395 | 14 % | 0.860 |

**Real search lists.** These are 3,583 releases from a media library's stored search results that
appear in no training file. Their cached mark came from that app's own checks, against more than
one service. AUC: 0.755. These lists are mostly cached releases (73 %), and every release in them
already passed the app's filters, so this is the harder test.

## Known limits

- **One service, one moment.** Caches expire and services differ. The number describes what
  TorBox held in November 2026, not what any cache holds today. It works as a worst-case estimate
  only for a service that caches at least as much as TorBox.
- **Small, popular encodes score well.** YTS/YIFY releases score a few points above other
  releases at the same resolution: 1080p 86 against 81, 720p 67 against 48. In the search lists
  above, a YTS release came first by availability in only 3 of the 65 lists that had one. A caller
  who cares about file quality should weigh availability by size per hour of running time and not
  use it alone.
- **A ranking hint, not a promise.** The middle of the range is uncertain by nature. Treat 0.5 as a
  coin toss.
- **The name only.** Two uploads of the same release with the same name get the same number, even
  if one of them is dead.

The training side, labels and scoring scripts are described in the training repository's
`cache_prediction/README.md`.
