# Live action: seven real names

Every name below is a real one, taken from the validation split of a large torrent index, not
written to make a point. Both columns are real output: the shipped NeuRelease model, and GuessIt
4.4.0 called with `name_only`, on the same string. The anime cases are in [ANIME.md](ANIME.md); the
aggregate numbers and the method are in [GUESSIT_COMPARISON.md](GUESSIT_COMPARISON.md).

## A translated title, in two languages at once

`Czas krwawego księżyca.Killers of the Flower Moon.2023.m1080p.WEBRip.H264.MP4.AC3-5.1 Lektor PL [StarLord]`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Killers of the Flower Moon | **Czas krwawego księżyca Killers of the Flower Moon** |
| alternative title | Czas krwawego księżyca | **m1080p** |
| year | 2023 | 2023 |
| release group | StarLord | StarLord |
| language | Polish | Polish |

A Polish distributor's title, then the English one, separated by a dot like every other field.
Fused into one title, the name matches nothing in any database. `m1080p` becoming an alternative
title is the same error from the other direction: the leftover has to go somewhere.

## A fused title in Polish and English, and two groups

`Uliczny Wojownik - Street Fighter 1994 [10Bit] [1080p.BluRay.H265-FT] [ENG-Lektor PL] [Alusia]`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Street Fighter | **Uliczny Wojownik** |
| alternative title | Uliczny Wojownik | Street Fighter |
| release group | FT, Alusia | Alusia |
| languages | English, Polish | English, Polish |

Both find two titles. NeuRelease elects the English one as the title and keeps the Polish as the
alternative, which is its documented policy; GuessIt takes whichever came first. It also finds both
groups: the encoder `FT` inside the bracket and the uploader `Alusia` at the end.

## Russian, with a season written in Russian

`Девять тел в мексиканском морге (Сезон 1) HDRezka`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Девять тел в мексиканском морге | Девять тел в мексиканском морге |
| season | 1 | 1 |
| pack | season | — |
| release group | HDRezka | — |
| episode title | — | **HDRezka** |

`Сезон 1` is read by both. The tracker name at the end becomes an episode title for GuessIt,
because something has to absorb a trailing word; NeuRelease reads it as the group and marks the
release a full-season pack.

## Eight seasons, stated twice

`Monk (2002) Season 1-8 S01-S08 (1080p AMZN WEB-DL x265 HEVC 10bit EAC3 Mixed RZeroX)`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title, year | Monk, 2002 | Monk, 2002 |
| seasons | 1–8 | 1–8 |
| pack | multi-season | — |
| release group | RZeroX | **Mixed RZeroX** |
| platform, source, codecs | AMZN, WEB-DL, HEVC, DDP, 10-bit | Amazon Prime, Web, H.265, Dolby Digital Plus, 10-bit |

The range is right on both sides, and the media facts agree. `Mixed` describes the audio, not the
group, and a group name with a word glued to the front matches nothing.

## A studio prefix and an indexer tag

`Marvels.Agents.of.S.H.I.E.L.D.S06E06.720p.HDTV.x264-AVS[ettv]`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Agents of S H I E L D | **Marvels Agents of S.H.I.E.L.D.** |
| franchise prefix | Marvels | not modelled |
| season, episode | 6, 6 | 6, 6 |
| release group | AVS | **AVS[ettv]** |

Two separate readings. The studio branding is not part of the title, and `ettv` is the indexer that
republished the file, not part of the group that made it. GuessIt keeps both glued on, which is
also one of its 22 documented limitations.

## British numbering, a broadcaster, and a part count

`BBC.Grand.Tours.of.Scotlands.Lochs.Series.4.2of6.1080p.HDTV.x264.AAC.MVGroup.org.mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Grand Tours of Scotlands Lochs | **BBC Grand Tours of Scotlands Lochs Series 4** |
| season | 4 | — |
| episode | 2 | 2 |
| broadcaster | BBC | — |
| release group | MVGroup.org | website MVGroup.org |
| pack | episode batch | episode count 6 |

`Series 4` is a season in British usage. Left inside the title, the title is wrong and the season
is missing, so nothing matches and nothing sorts.

## A dated episode, where GuessIt is right too

`Dateline.NBC.2026.05.08.Breaking.Point.480p.x264-mSD[EZTVx.to].mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Dateline | Dateline NBC |
| air date | 2026-05-08 | 2026-05-08 |
| episode title | Breaking Point | Breaking Point |
| broadcaster | NBC | — |
| numbering | date | date |
| release group | mSD | **mSD[EZTVx.to]** |

Both read the date correctly, which is the hard part of this name. The remaining differences are
conventions rather than errors: whether the broadcaster belongs in the title, and whether the
indexer tag belongs in the group.

## What this set does not show

These are names where the two disagree, selected from a corpus sweep for exactly that. On most
names they agree, which is why the headline is a percentage rather than a list: macro field F1 of
97.57% against 86.45%, and exact agreement on every shared field of 89.44% against 51.44%, measured
over 3,344 names in [GUESSIT_COMPARISON.md](GUESSIT_COMPARISON.md).

GuessIt also wins on its own regression corpus, 804 cases to 683.
