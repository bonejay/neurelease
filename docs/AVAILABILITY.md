# Availability

`availability` is one number from 0 to 1 per release name: **how likely this release is to be easy
to get**. It is read from the name alone, like everything else in the result. Size, age, seeders
and the title's popularity are not inputs.

```python
r = parser.parse("Paradise.2025.S01.1080p.10bit.WEBRip.6CH.x265.HEVC-PSA")
r.availability            # 0.89; None on a model without the head
```

```cpp
info.availability;        // 0.89F; -1 on a model without the head
```

```c
rp_availability(result);  /* 0.89f; -1 on a model without the head; also rp_result_view.availability */
```

## What it was learned from

The model has a small extra head trained next to the parser. Its label: did one debrid service,
TorBox, hold the torrent in its cache when asked? 176,763 real torrents were checked in November
2026, and 28 % were cached. Names with no real torrent behind them took no part, so a name counts
as unavailable only when the torrent was asked about and was not held.

A cache holds what people ask for and what stays seeded. The head therefore learns what wanted,
well-kept releases look like: established groups, complete season packs, current encodes. That is
why the number is called availability and not cache probability. It is useful for two jobs:

- **Ranking:** put likely-available releases first when nothing better is known.
- **A rough cache estimate:** before asking a debrid service, order the candidates by it. Treat it
  as a floor only for a service that caches at least as much as TorBox does.

It is **not a quality judgement.** In this library `screen_size`, `source` and `video_codec` say how
good the picture is; availability says how easy the release is to get. A small, very common encode
can score above a rare remux (see below).

## How well it works

All of these numbers come from names the model never trained on.

**Validation names.** 11,871 real torrents whose whole title family was kept out of training,
23 % cached, scored through the shipped int8 C++ runtime:

| | |
|---|---:|
| AUC (chance a cached name scores above an uncached one) | 0.883 |
| cached among the top tenth by availability | 81 % |
| cached among the bottom tenth | 0.3 % |
| Brier score | 0.110 |

The number reads roughly as a probability:

| availability | 0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| really cached | 5 % | 17 % | 23 % | 32 % | 42 % | 46 % | 52 % | 66 % | 78 % | 85 % |

It is not only a seeder count read from the name. The AUC holds within each seeder band:

| seeders | names | cached | AUC |
|---|---:|---:|---:|
| 50+ | 150 | 75 % | 0.934 |
| 5-49 | 3,398 | 39 % | 0.888 |
| 1-4 | 6,928 | 16 % | 0.884 |
| no swarm data (Anime Tosho) | 1,395 | 14 % | 0.860 |

**Real search lists.** These are 3,583 releases from a media library's stored search results that
appear in no training file. Their cached mark came from that app's own checks, from more than one
service. AUC: 0.755. These lists are mostly cached releases (73 %), which makes this the harder
test.

## Known limits

- **One service, one moment.** Caches expire and services differ. The number describes TorBox in
  November 2026, not any cache today.
- **Small, popular encodes score well.** YTS/YIFY releases score a few points above other releases
  at the same resolution: 1080p 86 against 81, 720p 67 against 48. They came first in only 3 of
  65 search lists that had one. A caller that cares about file quality should combine availability
  with size per hour of running time, not use it alone.
- **A ranking hint, not a promise.** Most of the probability is in the middle of the range. A
  release at 0.5 is cached about half the time.
- **Not every model has it.** `-1` in C and C++, and `None` in Python, mean that the installed model
  has no availability head. Never read that as "unlikely". Model 6 (2026-11-17) is the first with
  one.

The training side is described in the training repository's `cache_prediction/README.md`.
