# Anime

Anime naming came from fansubbing, not from the scene, and breaks most of what a rule-based parser
assumes: the group is written first in brackets, episodes are often absolute with no season, a work
carries an English and a romanised title at once, a dash may be a separator or part of the title,
and packs are advertised in prose (`(Season 1 & 2) + NC`, `[Batch]`, `01~13`).

Everything below is real output from the shipped model and from GuessIt 4.4.0 on the same string.
None of these names is in the labelled corpus at all, so none was trained on. Aggregates:
[GUESSIT_COMPARISON.md](GUESSIT_COMPARISON.md). Live action: [LIVE_ACTION.md](LIVE_ACTION.md).

## The anime verdict

A boolean on every result, **96.19% accurate** through the int8 runtime, the best-scoring of the
model's six classifications. It means animation from **Japan, China or Korea** - donghua and Korean
animation count; Western animation, live-action adaptations, tokusatsu and manga scans do not.

Earlier models answered whether a work was *animated*, which a name does not say:
`Shrek.2001.1080p.BluRay.x264` tells you nothing unless you already know what Shrek is. Anime a
name does announce, through the title, the group, the numbering and the tags. The form of the work,
`movie` or `series`, is the separate `content` field.

No other parser answers this, so it is left out of the tables below.

## Five names

### A season and an episode, read as a range

`[Golumpa] Re ZERO -Starting Life in Another World- Season 2 - 15 [CR-Dub 1080p x264].mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Re ZERO -Starting Life in Another World | Re ZERO |
| alternative title | — | Starting Life in Another World |
| season | 2 | **2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15** |
| episode | 15 | — |

Fourteen of those seasons do not exist.

### An alternative title in brackets, and a batch

`[Anime Time] Fire Force (Enen no Shouboutai) (Season 1 & 2) + NC [BD] [Dual Audio][1080p][HEVC 10bit x265][OPUS][Eng Sub] [Batch]`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Fire Force | Fire Force |
| alternative title | Enen no Shouboutai | — |
| season | 1–2 | 1, 2 |
| episode title | — | **NC** |
| release group | Anime Time | **Batch** |
| source, codec, audio | BluRay, HEVC, Dual Audio | Blu-ray, H.265, 10-bit, Opus, Dual Audio |

Both read the media facts. The difference is everything that identifies the work.

### A romanised title after the group

`Smoking Behind the Supermarket with You S01E07 The End of Summer Behind the Supermarket with You 1080p CR WEB-DL MULTi AAC2.0 H.264-VARYG (Super no Ura de Yani Suu Futari, Multi-Audio, Multi-Subs)`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Smoking Behind the Supermarket with You | same |
| alternative title | Super no Ura de Yani Suu Futari | — |
| season, episode, episode title | 1, 7, The End of Summer… | same |
| release group | VARYG | **VARYG (Super no Ura de Yani Suu Futari** |
| audio, subtitles | multi-audio, multi-subs | language mul, subtitle language mul |

A group with a title glued to it matches no group that exists.

### A subtitle tag that is not a language

`Go.For.It.Nakamura-kun.S01E04.The.Miraculous.Magical.Holy.Water.1080p.CR.WEB-DL.DUAL.DDP2.0.H.264.MSubs-ToonsHub.mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title, season, episode | Go For It Nakamura-kun, 1, 4 | same |
| release group | ToonsHub | **MSubs-ToonsHub** |
| subtitles | multi-subs, value `several` | — |
| audio, platform | dual audio, DDP 2.0, Crunchyroll | Dual Audio, Dolby Digital Plus 2.0, Crunchy Roll |

`MSubs` means several unnamed subtitle tracks, not the first half of a group name.

### A dash inside the title

`[HorribleSubs] Garo - Vanishing Line - 01 [1080p].mkv`

| | NeuRelease | GuessIt 4.4.0 |
|---|---|---|
| title | Garo - Vanishing Line | **Garo** |
| alternative title | — | **Vanishing Line** |
| numbering | absolute episode 1 | episode 1 |

The work is *Garo: Vanishing Line*. GuessIt's own issue #524, one of its 22 documented limitations.

## Related

| | |
|---|---|
| [LIVE_ACTION.md](LIVE_ACTION.md) | The same, for live-action names in four languages |
| [GUESSIT_COMPARISON.md](GUESSIT_COMPARISON.md) | Method, contract and per-field numbers |
| [RESULT.md](RESULT.md) | Every field, including `anime` and `absolute_episode` |
