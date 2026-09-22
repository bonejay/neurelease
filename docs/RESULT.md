# The result

Every field a parse can return, what it means, and how a release name states it.

Names are the Python ones, because most callers are Python callers. C++ spells the same field in
camelCase (`screen_size` is `screenSize`) and the C ABI in snake_case; where a name is shared with
GuessIt it is GuessIt's name. The C++ and C surfaces are in [API.md](API.md) and
[C_ABI.md](C_ABI.md).

Three rules hold everywhere, so they are stated once here rather than in every row:

- **A field the name does not state is `None`**, or an empty tuple for a list. Empty means "the
  name is silent", never a guessed default. In C++ that is `std::nullopt` or an empty
  string/vector, and in the C ABI `0` or NULL with a `stated` bit saying which zeros were written,
  since `S00` is a real season 0.
- **Closed fields are enums that equal their label**, so `r.source == "WEB-DL"` is true and
  `r.source.kind` is the member. Open fields are plain strings, kept as the name wrote them.
- **Every value carries `.confidence`**, the model's probability for the span behind it, or for
  the verdict in the case of the five whole-name decisions.

## Work identity

| Field | Type | Meaning | Stated as |
|---|---|---|---|
| `title` | `str` | The work's title as written, including its own subtitle. Never rewritten or canonicalised; separators become spaces. | `Blade Runner 2049`, `Kimetsu no Yaiba - Hashira Geiko-hen`, `葬送のフリーレン` |
| `alternative_title` | `tuple[str, ...]` | Other complete names of the same work in the same release name. `title` prefers English, then a romanisation, then the first native title. | `Shingeki no Kyojin` beside `Attack on Titan` |
| `franchise_prefix` | `str` | Branding written before the title that is not part of it. Model 3 reads one again, where the 2026-09-07 weights had stopped: on 205 labelled names carrying a prefix it returns prefix and title apart on 140, `James Bond 007` before `Casino Royale`. | `Marvels` in `Marvels.Agents.of.S.H.I.E.L.D`; `James Bond 007` |
| `episode_title` | `str` | The episode's own name, or that of a numbered film or special of a series. | `Der falsche Ranger`, `Chapter 19 The Convert` |
| `year` | `int` | The release or production year, when stated outside the title. | `2017`, `(2019)` |
| `date` | `AirDate` | Broadcast date, for releases numbered by date: news, daily shows, much adult content. Has `.year`, `.month`, `.day`. | `2026 08 25`, `2024-01-15` |

## Numbering

| Field | Type | Meaning | Stated as |
|---|---|---|---|
| `season`, `season_end` | `int` | The season, or the two ends of a season range. | `S02`, `Season 4`, `S01-08`, `2nd Season`, `Temporada 6` |
| `episode`, `episode_end` | `int` | Season-relative episode, or an episode range. | `E05`, `S01E01-E09`, `Cap.604`, `4x13` |
| `absolute_episode`, `absolute_episode_end` | `int` | Series-wide numbering, the anime convention. | `- 28`, `[01-12]`, `E150` with no season |
| `episode_count` | `int` | How many episodes a pack says it holds. | `12 Episodes`, `2of6` |
| `pack`, `complete_range` | `bool` | The release is a pack; the name states the range is complete. | `COMPLETE`, `Batch`, `01-12TV全集` |
| `specials` | `bool` | The release is or contains a special, OVA or movie entry. | `S00E01`, `OVA`, `特典映像` |

A name numbers its episodes one way, so `episode` and `absolute_episode` are never both set. Which
one it used is the `numbering` verdict below.

## Video

| Field | Type | Values or meaning | Stated as |
|---|---|---|---|
| `screen_size` | `ResolutionTier` | `480p` `720p` `1080p` `1440p` `2160p` `4320p` | `1080p`, `2160P`, bare `720`, `1920x1080` |
| `screen_size_text` | `str` | The resolution as the name wrote it, which is what `4K` or a bare `720` looked like before it became a tier. | `720`, `4K`, `1080p` |
| `frame_size` | `tuple[int, int]` | Exact dimensions, when the name gives them. | `1920x1080` → `(1920, 1080)` |
| `source` | `SourceKind` | `BluRay` `WEB-DL` `WEBRip` `HDTV` `DVD` `CAM` `Screener` `DCP` `Film` `WEB` `HDRip` | `BluRay`, `BDRip`, `BD`, `WEB-DL`, `WEBRip`, `HDTV`, `DVDRip`, `SCREENER`, `DCPRip`, `35MM.FilmScan` |
| `video_codec` | `VideoCodec` | `AV1` `HEVC` `H264` `XviD` `MPEG2` `VP9` `VC1` `WMV` `VVC` `VP8` `RealVideo` | `x265`, `HEVC`, `h264`, `AVC`, `264`, `XviD`, `RV10` |
| `streaming_service` | `str` | Service, channel or broadcaster. Open set. | `AMZN`, `NF`, `ATVP`, `BAHA`, `CR`, `DSNP` |
| `hdr` | `str` | The summary of the four flags below, else `SDR`. | `DV`, `HDR10+`, `HLG` |
| `hdr10`, `hdr10_plus`, `dolby_vision`, `hlg` | `bool` | The formats individually, since a release can carry several. | `HDR10`, `DV`, `HLG` |
| `ten_bit` | `bool` | Ten-bit video. | `10bit`, `Hi10P`, `HEVC-10bit` |
| `edition` | `tuple[EditionKind, ...]` | 61 kinds; see below | `EXTENDED`, `DC`, `UNRATED`, `IMAX`, `RESTORED`, `OAR`, `FANEDIT`, `初回限定版` |

A release is routinely several editions at once: `Uncut Unrated DC` is all three, so `edition` is a
tuple rather than one value.

The full set, in ABI order. A word the tables do not recognise is NOT forced into one of them: it
stays in `origins` as an edition span carrying its own text and no value, so a caller can tell
"read it and had no name for it" from "did not read it":

`IMAX` `Criterion` `Open Matte` `Remastered` `Unrated` `Uncut` `Uncensored` `Special Edition`
`Deluxe` `Redux` `Extended` `Director's Cut` `Final Cut` `Theatrical` `Despecialized`
`Assembly Cut` `Anniversary` `Signature` `Imperial` `Diamond` `2in1` `Preair` `Internal` `Limited`
`Untouched` `Dirfix` `Custom` `Widescreen` `Download` `Retail` `Collector` `Final` `Original`
`Fix` `Complete Edition` `Unabridged` `Re-encode` `Numbered Edition` `Regional` `High Quality`
`Ultimate` `Censored` `Fan Edit` `Bootleg` `Unofficial` `Bonus` `Festival` `Multi-Disc`
`Alternate Cut` `Shortened` `Leaked` `Colorized` `Fullscreen` `Standard` `Creditless`
`Re-recorded` `Commentary` `Explicit` `Reissue` `Original Aspect Ratio` `Restored`

Some of these draw distinctions that cost nothing to keep and are wrong to collapse.
`Final` is NOT `Final Cut`: French releases write `S01E08.FiNAL` for a season's last episode, and
routing that to a director's recut would be wrong on every one of them. `Restored` is not
`Remastered`: a restoration repairs damaged materials, a remaster re-derives from undamaged ones.
`Original Aspect Ratio` is not `Widescreen`: it says the transfer was not reframed, whatever shape
the frame is. `Censored` is the stated opposite of `Uncensored`, and both occur.

Two carry less than they appear to. `Numbered Edition` says a number was stated - `2ed`, `3rd
Edition` - without carrying which, because there is no edition-number field; the number stays
readable in the span text. `Regional` is the same compromise for `美版` and `japanische Fassung`: a
region-specific cut exists, without saying which region.

## Audio and subtitles

| Field | Type | Meaning | Stated as |
|---|---|---|---|
| `audio_codec` | `str` | The codec on its own. | `DDP`, `AAC`, `FLAC`, `TrueHD` |
| `audio_channels` | `str` | The channel layout, normalised to `N.M`. | `5.1`, `7.1`, `2.0` |
| `audio_profile` | `str` | The object-audio format layered over the codec. | `Atmos`, `DTS:X` (also written `DTS-X`, `DTSX`), `Auro-3D` |
| `language` | `tuple[str, ...]` | ISO 639-3 codes of the spoken languages the name states. | `German` → `deu`, `RUS` → `rus` |
| `subtitle_language` | `tuple[str, ...]` | ISO 639-3 codes of subtitle tracks. | `CHT` → `zho`, `Napisy PL` → `pol` |
| `subtitle_format` | `SubtitleFormat` | `SRT` `ASS` `VobSub` `PGS` `VTT` `SMI` | `SRT`, `ASS` |
| `dual_audio`, `multi_audio` | `bool` | Two, or more than two, audio languages. | `DL`, `Dual Audio`, `MULTI` |
| `multi_subs`, `hard_subs` | `bool` | Several subtitle tracks; subtitles burned into the picture. | `Multi-Subs`, `HardSub` |
| `english_dub` | `bool` | An English dub is included. | `English Dub`, `Dubbed` |

An empty `language` means the name did not say, not that the audio is English.

## Release identity

| Field | Type | Meaning | Stated as |
|---|---|---|---|
| `release_group` | `tuple[str, ...]` | The groups, in name order. Site banners and tracker tags are not groups and are kept out. | `TERMiNAL`, `SubsPlease`, `桜都字幕组`; not `[EZTVx.to]` or `[TGx]` |
| `container` | `str` | Lower case, no dot. | `mkv`, `mp4`, `epub`, `iso` |
| `proper`, `repack`, `remux` | `bool` | Re-release and remux markers. | `PROPER`, `REPACK`, `REMUX` |
| `release_version` | `int` | Which copy of the same release this is. `1` unless the name says otherwise. | `v2`, `REPACK`, `PROPER`, `REPACK2` |
| `release_real` | `int` | How many times the scene had to redo a botched re-release. Counted, not flagged. | `REAL`, `REAL.REAL.PROPER` |
| `hybrid` | `bool` | Two sources combined into one release; which two is not stated. | `Hybrid`, `HybridRip` |
| `light_encode`, `ai_upscale` | `bool` | Encode provenance. | `HDLight`, `VERSION_LIGHT`, `AI upscale`, `Topaz`, `AI增强` |
| `tokens` | `int` | How many tokens the model pooled the name to. Explains attention cost, not the release. | |

**The revision, and why it is one number.** A scene re-release says so in several ways at once and
they stack: `REPACK` is the second copy, `PROPER` is the second copy someone else made because the
first was broken, `v2` is the anime convention for the same thing, and `REPACK2` is the SECOND repack,
which makes it version 3. `release_version` is the single number to sort on - a proper or a repack
IS version 2, and a numbered one adds its number -
while `proper` and `repack` stay available for a caller that needs to know WHICH kind of re-release
it was. They are not alternatives: `PROPER.REPACK` sets both booleans and one version.

`release_real` is counted rather than flagged because the scene stacks the word: `REAL.PROPER` is
a fixed proper and `REAL.REAL.PROPER` the second attempt at fixing it. A boolean would read those
two as the same release.

**`hybrid` names no source, and that is the point.** `2160p.Hybrid.HDR10` says two sources were
combined and refuses to say which, so `source` stays unknown rather than guessing one of them. The
flag is the part that can be carried honestly, and it is raised whether the name puts the word
where a source belongs or among the editions.

## Whole-name verdicts

Six decisions no single substring can answer, so the model classifies the name as a whole. Each
carries its own confidence, which is what a caller filters on before acting.

`content` states the FORM and not the medium. Whether a work is animated is not something a release
name says - nothing in `Shrek.2001.1080p.BluRay.x265` reveals a cartoon - and a model asked anyway
answered animated films correctly 38% of the time while averaging 0.75 confidence when it was
wrong. So the medium is gone from this field, and `anime`, which a name does announce through its
group, its numbering and its title, is its own. An anime series is `content: series` with
`anime: true`.

| Field | Values | Example |
|---|---|---|
| `content` | `movie` `series` `music` `book_document` `comic_manga` `software` `game` `other` | `Marvels.Spider-Man.2018.PS4` → `game` |
| `anime` | `true` `false` | Japanese, Chinese and Korean animation, which the anime databases catalogue together; `[SubsPlease] Sousou no Frieren - 28` → `true`, `The.Simpsons.S27E05` → `false` |
| `medium` | `video` `music` `audiobook` `book` `comic` `software` `game` `image` `subtitle` `archive` | from the container and the content verdict together |
| `numbering` | `season_episode` `absolute` `volume` `date` `none` | `- 28` → `absolute`; `2026 08 25` → `date` |
| `pack_scope` | `single` `episode_batch` `volume` `season` `multi_season` `complete` | `S01E01-E09` → `episode_batch` |
| `special` | `none` `OVA` `special` `movie` | `S00E01` → `special`; `the.Movie` → `movie` |
| `adult` | `no` `yes` | |

`special` being `movie` marks a movie entry of a series, not that the release is a special. A named
season marked complete is `season`; `complete` needs evidence that the whole work is present.

## Evidence: `origins`

`origins` holds every located fact in name order, and is how any field can be traced back to the
bytes it came from. Alternative titles, the franchise prefix, the episode title, site banners and
tracker tags all live here too.

| Member | Meaning |
|---|---|
| `field` | which fact this is (`TITLE`, `ALTERNATE_TITLE`, `FRANCHISE_PREFIX`, `EPISODE_TITLE`, `YEAR`, `SEASON`, `QUALITY`, `SOURCE`, `CODEC`, `GROUP`, `SITE_BANNER`, …) |
| `value` | the canonical value, the same one the flat field holds |
| `text` | the raw substring that produced it, verbatim |
| `begin`, `end` | that substring's UTF-8 byte range in the name |
| `confidence` | the model's probability for this span |
| `unconverted` | located and typed, but no alias table entry for its text |

Spans never overlap and keep source order. Evidence is opt-in, since building it costs a little:
`parser.parse(name, origins=True)`.

```python
for o in parser.parse("Gladiator.EXTENDED.2000.720.BrRip.264.YIFY", origins=True).origins:
    if o.begin >= 0:   # the whole-name verdicts are in the same list, without a span
        print(f"{o.field.name:<8} {o.text:<9} -> {o.value:<9} bytes {o.begin}-{o.end}  {o.confidence:.2f}")
```

```text
TITLE    Gladiator -> Gladiator bytes 0-9    1.00
EDITION  EXTENDED  -> Extended  bytes 10-18  1.00
YEAR     2000      -> 2000      bytes 19-23  1.00
QUALITY  720       -> 720p      bytes 24-27  1.00
SOURCE   BrRip     -> BluRay    bytes 28-33  1.00
CODEC    264       -> H.264     bytes 34-37  0.98
GROUP    YIFY      -> YIFY      bytes 38-42  0.95
```

`unconverted` is what separates "not stated" from "stated but not understood". A Chinese subtitle
marker missing from the alias tables is still located, typed and kept out of the title, and it is
reported here rather than dropped. Adding the spelling to `data/aliases/` is the fix.

## When a parse goes wrong

`valid` is false when nothing could be read at all. `degraded` is true when the model could not
encode the name and only the title and year heuristics ran; it is rare, and it is surfaced so a
ranker can trust that parse less.

## The plain dict

`to_dict()` returns the stated facts as strings, ints, lists and booleans, under GuessIt's property
names wherever GuessIt answers the same question, so code written against GuessIt reads it
unchanged. `confidence` maps each emitted key to its number. `to_dict(origins=True)` appends the
evidence. The shape is documented in [API.md](API.md).
