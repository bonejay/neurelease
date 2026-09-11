# Anime

Anime naming is not scene naming. The conventions came from fansubbing rather than from the
release groups whose patterns most parsers were written against, and almost every one of them
breaks an assumption those parsers make:

- The group is written **first**, in brackets, not last after a dash.
- Episodes are often **absolute**, with no season anywhere in the name.
- A work carries **two titles at once**, English and romanised, and either can come first.
- A dash can be a separator, or part of the title, or both in one name.
- Batches, season packs and non-credit openings are advertised in prose: `(Season 1 & 2) + NC`,
  `[Batch]`, `01~13`.
- The audio and subtitle situation is a tag rather than a language: `Dual Audio`, `MSubs`,
  `Multi-Subs`, `Eng Sub`.

This document is what NeuRelease does with all of that, and what a rule-based parser does beside
it. Everything below is real output, produced by the shipped model and by GuessIt 4.4.0 on the same
strings.

## The anime verdict

`anime` is a boolean on every result, with its own confidence. It means **animation from Japan,
China or Korea** - the animation the anime databases catalogue. Chinese donghua and Korean
animation count, because they are made, released and numbered the same way and those databases
list them beside Japanese works. Everything else is `false`, including Western animation however
anime-influenced, live-action adaptations, tokusatsu, and manga or light novel releases.

It reads at **96.50% accuracy** through the int8 runtime on the validation split, and it is the
best-scoring of the model's six classifications.

Earlier models answered a different question - whether a work was *animated* - and that question
turned out to be unanswerable from a name. `Shrek.2001.1080p.BluRay.x264` says nothing about
animation; you have to already know what Shrek is. Whether something is anime, a name does
announce, through the title, the group, the numbering and the tags. So model 3 dropped the
guess and kept the readable question. The form of the work, `movie` or `series`, became a separate
field, which is what `content` now carries.

## Seven names, both parsers

### Season and episode written as a range

`[Golumpa] Re ZERO -Starting Life in Another World- Season 2 - 15 [CR-Dub 1080p x264].mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Re ZERO -Starting Life in Another World | Re ZERO |
| alternative title | — | Starting Life in Another World |
| season | 2 | **2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15** |
| episode | 15 | — |
| anime | true | not answered |

`Season 2 - 15` is a season and an episode. Read as a range it produces fourteen seasons that do
not exist. The dashes around the stylised subtitle are also part of the title here, not separators,
which is why the title should not split at them.

### An alternative title in brackets, and a batch

`[Anime Time] Fire Force (Enen no Shouboutai) (Season 1 & 2) + NC [BD] [Dual Audio][1080p][HEVC 10bit x265][OPUS][Eng Sub] [Batch]`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Fire Force | Fire Force |
| alternative title | Enen no Shouboutai | — |
| season | 1–2 | 1, 2 |
| pack | multi-season | — |
| episode title | — | **NC** |
| release group | Anime Time | **Batch** |
| source, codec, audio | BluRay, HEVC, Dual Audio | Blu-ray, H.265, 10-bit, Opus, Dual Audio |
| anime | true | not answered |

Both read the media facts. The difference is everything that identifies the work: the romanised
title is lost, `NC` (non-credit opening and ending) becomes an episode title, and the group becomes
the word `Batch`.

### A romanised title after the group

`Smoking Behind the Supermarket with You S01E07 The End of Summer Behind the Supermarket with You 1080p CR WEB-DL MULTi AAC2.0 H.264-VARYG (Super no Ura de Yani Suu Futari, Multi-Audio, Multi-Subs)`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Smoking Behind the Supermarket with You | Smoking Behind the Supermarket with You |
| alternative title | Super no Ura de Yani Suu Futari | — |
| season, episode | 1, 7 | 1, 7 |
| episode title | The End of Summer Behind the Supermarket with You | same |
| release group | VARYG | **VARYG (Super no Ura de Yani Suu Futari** |
| audio, subtitles | multi-audio, multi-subs | language mul, subtitle language mul |
| anime | true | not answered |

The group swallows the romanised title, so it matches no group that exists and the second title is
gone. Both parsers read the season, the episode and the episode title correctly, which are the hard
parts of this name; the group boundary is where the conventions differ.

### A subtitle tag that is not a language

`Go.For.It.Nakamura-kun.S01E04.The.Miraculous.Magical.Holy.Water.1080p.CR.WEB-DL.DUAL.DDP2.0.H.264.MSubs-ToonsHub.mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title, season, episode | Go For It Nakamura-kun, 1, 4 | same |
| release group | ToonsHub | **MSubs-ToonsHub** |
| subtitles | multi-subs, value `several` | — |
| audio | dual audio, DDP 2.0 | Dual Audio, Dolby Digital Plus 2.0 |
| platform | Crunchyroll | Crunchy Roll |
| anime | true | not answered |

`MSubs` means several subtitle tracks, none of them named. NeuRelease records it as a subtitle
language whose value is `several` and sets the `multi_subs` flag; GuessIt reads it as the first
half of the group name.

### Absolute numbering with no season

`[SubsPlease] Sousou no Frieren - 28 (1080p) [F02B9CEB].mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Sousou no Frieren | Sousou no Frieren |
| numbering | absolute episode 28 | episode 28 |
| release group | SubsPlease | SubsPlease |
| crc32 | — | **F02B9CEB** |
| anime | true | not answered |

Agreement, with one distinction: absolute numbering is a different fact from a season-relative
episode, and a library that treats episode 28 as season 1 episode 28 will look for the wrong file.
GuessIt reads the CRC32 here and NeuRelease does not.

### A dash inside the title

`[HorribleSubs] Garo - Vanishing Line - 01 [1080p].mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Garo - Vanishing Line | **Garo** |
| alternative title | — | **Vanishing Line** |
| numbering | absolute episode 1 | episode 1 |
| anime | true | not answered |

The work is called *Garo: Vanishing Line*. This is GuessIt's own documented limitation, issue
#524, and one of the 22 cases in its known-limitations page.

### Donghua

`Douluo.Dalu.Soul.Land.S01E250.1080p.WEB-DL.AAC.H264-Lamb`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title, season, episode | Douluo Dalu Soul Land, 1, 250 | same |
| source, codecs | WEB-DL, H264, AAC | Web, H.264, AAC |
| anime | **true** | not answered |

Identical, apart from the classification. This is Chinese animation in a scene-shaped name, and
`anime: true` is a fact no rule-based parser answers at all.

## What NeuRelease gets wrong here

Measured on the names above rather than claimed:

- On the Smoking Behind name it returns `pack_scope: season` for a single stated episode. The name
  says `S01E07` and advertises no pack.
- On the Re:Zero name the title keeps the leading dash of the stylised subtitle while dropping the
  trailing one: `Re ZERO -Starting Life in Another World`.
- It does not read the CRC32 on the Frieren name, which GuessIt does.

And on GuessIt's own regression corpus, GuessIt wins 804 cases to 683. That corpus is its test
suite, written to exercise its rules; the method and the per-field numbers are in
[GUESSIT_COMPARISON.md](GUESSIT_COMPARISON.md).

## Related

| | |
|---|---|
| [GUESSIT_COMPARISON.md](GUESSIT_COMPARISON.md) | The full comparison: method, contract, per-field numbers |
| [RESULT.md](RESULT.md) | Every field a result carries, including `anime` and `absolute_episode` |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The model, and what changed in version 3 |
