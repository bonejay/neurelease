# Using the parser

The three surfaces carry different amounts of detail, by design:

| Surface | What you get |
|---|---|
| **C++** | everything: `ReleaseInfo`, the full `Analysis` (complete segmentation, span probabilities, token count, kernel path), `BatchParser::lastTiming()` per-stage timings |
| **C ABI** | `ReleaseInfo` flat fields, verdicts with confidences, air date, and the origin spans — the stable subset every binding builds on |
| **Python** | the flat fields and verdicts by default; the origin spans on request (`origins=True`). `Analysis` and timings stay native — converting them per name would cost more than the parse |

If you need the full diagnostics — per-stage timings, the complete segmentation, which SIMD kernel
ran — use the C++ API.

The four model files are committed under `model/`, so nothing needs downloading and the Python
binding finds them on its own. C++ and C still take the directory explicitly — a library installed
system-wide has no business guessing where model files live.

### Python

A small `ctypes` wrapper over the C ABI. No second parser, no compiled extension.

```python
from neurelease import Parser, SourceKind

with Parser() as parser:
    r = parser.parse("Power.Rangers.Megaforce.S01E04.Der.falsche.Ranger.German.DL.720p.BluRay.x264-TV4A")
    r.title, r.episode_title            # 'Power Rangers Megaforce', 'Der falsche Ranger'
    r.season, r.episode, r.year         # 1, 4, None  (None = the name does not say; S00 is a real 0)
    r.source == SourceKind.BLURAY       # True; r.source.kind is the enum member
    r.source == "BluRay"                # also True: closed values are their label
    r.title.confidence                  # 0.73: every value carries the model's confidence
    r.screen_size, r.screen_size_text   # ResolutionTier.P720 (== "720p"), '720p' as written
    r.to_dict()                         # the stated facts as plain values (below)

    results = parser.parse_batch(["Show.S01E01.1080p.WEB-DL.x264-GRP.mkv",
                                  "Show.S01E02.1080p.WEB-DL.x264-GRP.mkv"])
```

**The result object.** `ParsedRelease` is a frozen dataclass. Its fields carry GuessIt's
property names wherever the fact is the same, singular for the tuples as GuessIt has them, and
our own names only for facts GuessIt lacks. Open facts are strings or tuples (`title`,
`episode_title`, `alternative_title`, `franchise_prefix`, `streaming_service`, `audio_codec`,
`audio_channels`, `audio_profile`, `release_group`, `language`, `subtitle_language`, `container`),
numbers are `int | None`, and closed facts are closed values: `screen_size`, `source`,
`video_codec`, `medium`, `content`, `numbering`, `pack_scope`, `special`, `adult`,
`subtitle_format`, `edition`. Each is a `Value`: a `str` holding the label
(`"BluRay"`, `"1080p"`, `"series"`) that compares equal to the enum member
(`r.source == SourceKind.BLURAY`) and to the label, with `.kind` for the member itself, `.name`,
`.id` (the C ABI integer) and `.confidence`. `screen_size_text` and
`frame_size` (`(1920, 1080)` when the name states dimensions) come from the evidence spans, so they
are `None` with `origins=False`; `episode_title`, `alternative_title` and `franchise_prefix` are
fields of the result itself.

**Confidence.** Every value carries its own: `r.title.confidence`, `r.season.confidence`,
`r.release_group[0].confidence`, `r.content.confidence`. For a located field it is the lowest span
probability behind the value; for a verdict, the verdict's probability. Filter on the value you
care about: `if r.title.confidence < 0.9`. The values are otherwise plain: `Text` is a `str`, `Number` an `int`,
`Value` a `str` label with `.kind` (the enum member), `.name` and `.id`, `Items` a tuple.

**`to_dict(origins=False)`.** The stated facts as plain values, under GuessIt's property names
wherever GuessIt has the same fact, so code written against GuessIt reads it unchanged: `title`,
`alternative_title`, `episode_title`, `year`, `date`, `season`, `episode`, `absolute_episode`,
`episode_count`, `screen_size`, `source`, `video_codec`, `audio_codec`, `audio_channels`,
`audio_profile`, `color_depth`, `streaming_service`, `edition`, `other`, `language`,
`subtitle_language`, `release_group`, `container`, `crc32`, `website`, `type`. As in GuessIt, a fact
with one value is a scalar and with several a list, and a range is the list of its numbers
(`"episode": [1, 2, 3]`). Facts GuessIt has no name for keep ours: `franchise_prefix`, `content`,
`medium`, `adult`, `numbering`, `pack_scope`, `special`, `hdr`, `frame_size`. Values stay ours
(`WEB-DL`, `HEVC`, ISO 639-3 codes), unstated facts are absent rather than `None`, and
`confidence` maps every emitted key to its number. `origins=True` adds the evidence spans as a
list of dicts. About 8 microseconds per name on top of the parse.

```python
{'title': 'Seihantai na Kimi to Boku', 'season': 2, 'episode': 4, 'screen_size': '1080p',
 'release_group': 'Erai-raws', 'container': 'mkv', 'type': 'episode',
 'content': 'series', 'anime': True, 'medium': 'video', 'adult': False, 'numbering': 'season_episode',
 'confidence': {'title': 0.998, 'season': 0.997, 'episode': 0.999, ...}}
```

`Parser()` looks for the models beside the package, then in the repository's `model/`, then in the
working directory; `Parser("some/dir")` or `NEURELEASE_MODELS` overrides that. The shared library
is found the same way, or named with `library=` / `NEURELEASE_LIBRARY`. Put `bindings/python` on
`PYTHONPATH`, or package that directory normally.

Every located fact keeps the text and byte range it came from:

```python
result = parser.parse(name, origins=True)   # evidence is opt-in
for origin in result.origins:
    print(origin.field, origin.text, origin.begin, origin.end, origin.confidence)
```

### C++

The C++ API is the full result: `ReleaseInfo` with typed enums, `std::optional<int>` for every
number the name may not state, `std::vector<std::string>` for the lists, and `Analysis` with the
complete segmentation. No dict view and no label strings; those are Python conveniences over the
same facts.

```cpp
#include <neurelease/parser.hpp>

neurelease::Parser parser("model");
const neurelease::ReleaseInfo info = parser.parse(
    "Power.Rangers.Megaforce.S01E04.Der.falsche.Ranger.German.DL.720p.BluRay.x264-TV4A").info;

info.title;                         // "Power Rangers Megaforce"
info.episodeTitle;                  // "Der falsche Ranger"
info.season, info.episode;          // std::optional<int>: 1, 4
info.year.has_value();              // false: the name does not say
info.source == neurelease::SourceKind::BluRay;   // enums for every closed value
neurelease::label(info.source);     // "BluRay", when text is wanted
info.languages;                     // {"deu"}; info.dualAudio is true
info.content;                       // ContentKind::Series, info.contentConfidence 0.90
info.anime;                         // false, info.animeConfidence 0.97 - the form and the
                                    // tradition are separate answers, see docs/RESULT.md

for (const auto& origin : info.origins)     // every located fact with its bytes
    std::cout << neurelease::fieldName(origin.field) << " <- " << origin.text
              << " [" << origin.begin << ", " << origin.end << ")
";
```

For an installed package:

```cmake
find_package(neurelease CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE neurelease::core)
```

Parse naturally grouped inputs together:

```cpp
neurelease::BatchParser parser("model");
std::vector<std::string> names = load_names();
std::vector<neurelease::ParseResult> results = parser.parse(names);
```

Results preserve input order. A parser instance is used by one caller at a time;
`BatchParser` owns its worker threads.

### C ABI

The stable C API is [include/neurelease/release_parser.h](../include/neurelease/release_parser.h);
its rules and version history are in [C_ABI.md](C_ABI.md).

```c
#include <neurelease/release_parser.h>

rp_parser* parser = NULL;
rp_result* result = NULL;

if (rp_parser_new("model", &parser) == RP_OK &&
    rp_parse(parser, "Show.S01E03.1080p.WEB-DL.HEVC-GRP.mkv", &result) == RP_OK) {
    const char* title = rp_str(result, RP_FIELD_TITLE); /* borrowed */
    rp_resolution_tier screen_size = rp_screen_size(result);
    rp_content_kind content = rp_content(result);
}

rp_result_free(result);
rp_parser_free(parser);
```

This is the language-neutral binary contract behind Python, and the one to bind from Rust, C#, Go,
Node or Java. Converting a completed parse into an application's own domain objects stays the
application's job.

## Options

**Batch worker threads.** The default is half the logical cores. In C++ pass it to the
constructor; through the C ABI (and therefore before Python's `parse_batch`) set it explicitly:

```cpp
neurelease::BatchParser parser("model", /*threads=*/4);
parser.setBucketSize(64);   // names per length-sorted bucket; smaller buckets, less padding
```

```python
with Parser(threads=4) as parser:
    results = parser.parse_batch(names)
```

```c
rp_set_batch_threads(parser, 4);   /* 0 restores the default; rebuilds the batch engine */
```

**Model and library discovery** (Python): `NEURELEASE_MODELS` names the model directory,
`NEURELEASE_LIBRARY` the shared library; both beat the built-in search. C++ and C always take
the model directory as an argument.

**Everything by default.** `parse` and `parse_batch` include the origin spans — each located fact
with its raw text, byte range and per-span confidence. They cost about 6% on a large batch;
`origins=False` skips them when that matters, and with them `screen_size_text` and `frame_size`.
The flat fields, verdict confidences and air date are always present. `tools/bench_python.py` measures
all of this on your machine.

**Kernel accuracy.** On AVX2-only CPUs the fastest int8 kernel is not bit-exact — about 1% of
names parse differently, wins and losses balancing out. `NEURELEASE_SEGMENTER_ACCURACY=exact`
restores the exact kernels for a session. VNNI and AVX-512 machines are exact either way.

**Timing.** `BatchParser::lastTiming()` reports wall time, per-stage model time, worker count and
bucket padding for the previous call — the numbers behind the benchmark.

