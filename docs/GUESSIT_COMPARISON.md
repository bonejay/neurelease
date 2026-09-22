# Comparison with GuessIt 4.4.0

Rerun on 2026-09-22 using the installed **model version 5**, retained epoch 19 from
`corrected/20260921-233603` - the first model trained after the corpus repairs: the DeepSeek second
pass and its adjudication over every row that rested on a low-effort label, the settled `Part N`
convention, the walk relabel, and the split that keeps every scored name out of training. The
model SHA-256 is `7857dcdb813fac941198b4243384f51a91d291b4022258501c5317092d932c34`.

The [scoring snapshot](SCORING_SNAPSHOT.json) records the model, dataset, library and scoring-source
hashes, repository revisions, raw timing samples and per-field scores. The build was brought up to
date with `cmake --build --preset release` before evaluation, which for this run mattered: the
weights and the conversion layer both changed with the new form and anime fields.

The headline table is in the [README](../README.md); this document is the method behind it.
Name-by-name readings, where a difference can be seen rather than averaged, are in
[ANIME.md](ANIME.md) and [LIVE_ACTION.md](LIVE_ACTION.md).

## How accuracy is scored

Both parsers read the same 3,344 video names from `tests/data/consensus_validation.jsonl`.
The labels are LLM-written, with two passes in agreement. This is a development validation set
used for checkpoint selection, not an independent final test. Music, software, games and other
non-video names are excluded from this comparison because they are outside GuessIt's domain.

NeuRelease runs through the current Python binding and C ABI with the committed int8 weights;
GuessIt 4.4.0 runs through `GuessItApi` with `name_only=True`, `no_user_config=True` and
`enforce_list=True`. The comparison adapters normalize their outputs into the same 20-field
contract before comparing against gold. Content classification and distinct main/alternate-title
selection are diagnostics and do not contribute to these headline scores.

- **Exact on every applicable shared field:** the share of names for which all applicable
  compared value sets match gold. A missing or extra value makes that field fail.
- **Macro shared-field F1:** precision and recall are calculated per field over its applicable
  names, then the field F1 scores are averaged with equal weight across fields having gold support.
- **Applicability:** `work_title` is scored only where gold supplies one title representation;
  other headline fields follow their recorded contract. This is not an all-output-fields score.
- **Normalization:** case and punctuation are normalized, aliases share a vocabulary, source
  variants map to coarse families, season/episode ranges are projected consistently, and absolute
  episodes answer the same shared episode question. Moving a number from `episode` to
  `absolute_episode` alone cannot change that score. Tracker-tag/group spelling differences are
  resolved from returned values.
- **Subtitles:** `SEGMENTER_SUBTITLES=merge` is set explicitly, matching the shipped model's
  training convention. Gold work-subtitle spans are folded into their titles and predictions are
  read under the same convention.

## By field

The name count is gold support, not necessarily the full set of names evaluated for false
positives. Full evaluated counts, precision, recall and exact-present rates are in the snapshot.

| Field | Names with it | NeuRelease F1 | GuessIt F1 |
|---|---:|---:|---:|
| `work_title` | 2,997 | 97.42% | 90.97% |
| `episode_title` | 173 | 91.57% | 73.66% |
| `year` | 1,175 | 99.74% | 98.68% |
| `season` | 1,227 | 99.15% | 91.30% |
| `season_end` | 25 | 95.83% | 82.14% |
| `episode` | 1,280 | 97.86% | 90.70% |
| `episode_end` | 130 | 95.38% | 82.87% |
| `resolution` | 2,372 | 99.81% | 99.73% |
| `source_family` | 1,913 | 98.61% | 96.95% |
| `platform` | 415 | 98.31% | 69.45% |
| `video_codec` | 1,818 | 99.23% | 98.58% |
| `audio_codec` | 927 | 99.12% | 97.73% |
| `audio_channels` | 550 | 99.73% | 99.46% |
| `audio_language` | 265 | 97.01% | 66.31% |
| `subtitle_language` | 154 | 95.76% | 45.81% |
| `bit_depth` | 253 | 98.64% | 97.86% |
| `container` | 999 | 99.85% | 99.50% |
| `crc32` | 164 | 100.00% | 98.80% |
| `release_group` | 2,320 | 99.28% | 69.76% |
| `release_variant` | 125 | 95.37% | 84.58% |

## External case sets

On GuessIt's regression corpus NeuRelease passes **693/859** cases (4,386/4,613 field assertions),
GuessIt **804/859** (4,552/4,613). The corpus is GuessIt's own test suite: fixture strings such as
`FooBar.307.PDTV-FlexGet` and filesystem paths, written to exercise its rules, with every input
assumed to be a video. NeuRelease parses a single release name and classifies it before assuming
anything, so the shapes it misses here are largely the fixtures and conventions, not real names.
Scored are the 859 cases left after excluding 165 path entries, non-equivalent content-type checks,
unsupported vocabulary and documented incorrect expectations; the snapshot records every exclusion,
and only fields a case explicitly states are checked.

The native `guessit_corpus_tests` uses a stricter title contract and a different denominator; its
score must not be substituted for the comparison harness's 859-case score.

On the 22 cases from GuessIt's documented limitations, NeuRelease passes **21/22** cases and
**71/72** assertions; GuessIt passes **0/22** and **31/72**. These are selected failure cases for
GuessIt, not a representative accuracy sample. They are scored separately from validation.

## Timing

Single-name timings are medians of three complete warm loops over the same 3,344 names on an
AMD Ryzen 7 5700X3D (8 cores / 16 logical processors). They include the Python calls and each
comparison adapter's field projection. Initialization and the first parse are measured separately
and excluded from the warm-loop figures. Each parser is measured sequentially.

The four-worker row is a separate measurement of `Parser(..., threads=4).parse_batch(names)`:
three passes, 64 warm-up names, origins enabled, full Python results materialized, no comparison
projection. The median is shown. The comparison harness's own batch timer uses the default of
half the logical processors (eight on this machine); its samples are retained in the snapshot
but are not mislabeled as four-worker results. Timings vary with machine load and boost clocks.

## Reproduce

Run from the sibling training repository after building the inference library:

```powershell
$env:SEGMENTER_SUBTITLES = 'merge'
.\.venv-comparison\Scripts\python.exe training/compare_parsers.py `
    --labels ../neurelease/tests/data/consensus_validation.jsonl `
    --neurelease-library ../neurelease/build/release/libneurelease.dll `
    --neurelease-model ../neurelease/model `
    --repetitions 3 `
    --json-out build/readme-current-score/comparison.json `
    --markdown-out build/readme-current-score/comparison.md `
    --cases-out build/readme-current-score/cases.jsonl
```

The full generated JSON and per-name JSONL stay in the training workspace; the compact published
snapshot carries the evidence needed to identify this run. The local four-worker timing script is
`build/readme-current-score/bench_four_workers.py`; its settings and all samples are in the snapshot.
The scoring code lives under `training/parser_comparison/` in the training repository.
