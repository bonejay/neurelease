# Architecture

`neurelease` contains learned inference only. Training, corpora, gold evaluation, and the
comparison harness live in the training repository.

## Data flow

1. The UTF-8 preprocessor decodes a bounded name, identifies Chinese/Japanese text, applies
   contextual transliteration, and produces source-aligned character features.
2. The segmenter runs the character convolution stack, pools pseudo-tokens, runs the token
   encoder, and emits typed spans plus five whole-name verdicts. This is the only learned step.
3. The schema boundary converts the exported class labels into enums once, at load. Construction
   fails if a model adds, removes, reorders, or duplicates a class the runtime does not know.
4. The mapper turns the text of each typed span into a canonical value (`x265` → `VideoCodec::Hevc`,
   `2160P` → `ResolutionTier::P2160`, `E-AC-3.5.1.Atmos` → structured audio) and assembles
   `ReleaseInfo`. It never scans the name with a second parsing engine.
5. `Parser` returns one result. `BatchParser` owns a parser per worker, schedules length-aware
   buckets, and restores input order.
6. The C ABI exposes the same result through opaque handles. Language bindings copy borrowed views
   before freeing their owner.

Closed values are enums (span and verdict classes, screen size, source, codec, medium, subtitle
format, edition, origin field); open values stay UTF-8 strings (titles, release groups,
streaming services, containers, language codes, audio codec, channels and profile). The consuming application owns everything after
`ReleaseInfo`: provider identity, ranking, swarm state, requested-title repair, UI types.

## Model pipeline

```text
UTF-8 release name
    -> Unicode decoding and source alignment
    -> Han-script routing by a quantized hashed linear SVM
    -> contextual Chinese/Japanese transliteration (phrase tries)
    -> character, script, and case embeddings
    -> three-layer dilated character CNN
    -> letter/digit-run pooling into pseudo-tokens
    -> six-layer token Transformer + five summary tokens
    -> boundary head + 35-way span head + five whole-name heads
    -> typed conversion -> ReleaseInfo + Analysis
```

**Preprocessing.** Decoding keeps the original byte location of every codepoint. Kana settles the
language on its own; a Han-only name goes through a filename-level classifier, a linear SVM over
hashed character n-grams with int16 weights (65 KB, 97.9% balanced accuracy; logistic regression
scored 97.8%, naive Bayes 96.4%). The chosen phrase trie emits Latin-letter features aligned to
every original character, so titles and evidence always carry the untouched input text.

**Character CNN and pseudo-tokens.** Each position combines 104 character, 16 script, and 8
case/boundary features. Three width-5 convolutions (128, 128, 144 channels; dilations 1, 1, 2) read
structured forms and punctuation context without losing alignment. Letter and digit runs are then
max-pooled into 144-wide pseudo-tokens, roughly 8 to 20 per name, so attention works on the semantic
sequence rather than on every character. `Analysis::tokens` exposes the count.

**Token Transformer and heads.** Six encoder layers, 256 wide, 8 heads of 32, FFN 512, plus five
whole-name summary tokens. A boundary head finds where spans begin, a 35-way type head labels them,
and the summary tokens produce the five verdicts independently.

**Conversion.** Open vocabularies (platform and language aliases, containers, rejected group tokens)
are JSON under `data/aliases/`; composition and closed enum conversion are typed C++. The build
compiles the JSON into sorted immutable arrays when Python is available (edit the JSON, build,
done); without Python the committed header under `src/convert/generated/` is used as-is. Regenerate
and commit it when a vocabulary change is permanent:

```sh
python tools/generate_aliases.py           # refresh the committed copy
python tools/generate_aliases.py --check   # verify it matches the JSON
```

Inference parses no JSON and builds no hash tables at startup.

## Model artifacts

| File | Purpose | Size |
|---|---|---:|
| `segmenter.bin` | row-wise int8 CNN, Transformer, heads, and class schema | 4,021,856 bytes |
| `chinese_japanese.bin` | packed classifier for ambiguous Han text | 65,572 bytes |
| `transliteration_japanese.bin` | contextual Japanese phrase trie | 751,108 bytes |
| `transliteration_chinese.bin` | contextual Chinese phrase trie | 238,069 bytes |

The transliteration artifacts are longest-match phrase tries, not dictionaries or neural models:
they supply compound-aware Japanese and phrase/polyphonic Chinese readings as aligned features and
never replace the original title. All four load directly from Git; `model/manifest.json` records
format versions, sizes, and SHA-256 digests, and says which training run the segmenter came from; each loader verifies its own file's format header, and the training repository's install script rewrites the manifest whenever it ships new weights.
Large Unicode transliteration tables are compiled into the library as compact sorted arrays.

## Model versions

`model_version` in `model/manifest.json` counts retrains of the segmenter; the other four files
have not changed since version 1 and keep their own `format_version`. The corpus score is the
native `guessit_corpus_tests` floor, cases fully correct of 858. An older version is
`git show <tag>:model/segmenter.bin`.

| Version | Trained | Training run | Corpus | What changed |
|---:|---|---|---:|---|
| 1 | 2026-09-03 | `merge-sub/20260903-004552` | 652 | a work's subtitle folded into its title, the one span boundary two labelling passes never agreed on |
| 2 | 2026-09-07 | `hybrid-labelled/20260907-002017` | 667 | trained on 6,479 labelled `Temporada 6 [Cap.604]` names: 14 of 16 such corpus cases against 5; the franchise prefix is no longer split from the title and some episode titles split in two |
| 3 | 2026-09-11 | `anime2/20260911-032100` | 663 | the animated/live-action content kinds became a `movie`/`series` form plus a separate `anime` verdict, which the name can actually be read for: 96.50% accuracy through the int8 runtime. Four fixture cases traded for a better reading of real names (683 of 859 in the comparison harness against 682, exact agreement 89.44% against 89.08%) |

## Performance rules

- CPU dispatch is selected once; hot loops do not use virtual dispatch.
- ISA-specific sources are compiled with only their required flags and selected after CPU checks.
  The int8 kernels are self-checked against the scalar path at first use and demoted to it,
  persistently, if anything disagrees.
- Model weights and parser scratch are reused; the public batch call owns parallelism, and callers
  must not share one parser concurrently.
- Diagnostics and timing live beside the semantic result so equality is deterministic.

## Build

Requirements: CMake 3.24+, a C++23 compiler, and Ninja or another CMake generator. PCRE2 is the
only third-party library dependency and is fetched automatically when not installed.

```sh
git clone https://github.com/bonejay/neurelease.git
cd neurelease
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
python -m pytest bindings/python/tests
```

`cmake --install build/release --prefix dist` produces a self-contained package. To use the Python
binding from a source build rather than from PyPI, `pip install ./bindings/python` afterwards.

### Options

Options: `RP_BUILD_C_API`
(default on), `RP_BUILD_TESTS` (on), `RP_BUILD_BENCHMARKS` (off), `RP_FETCH_DEPENDENCIES` (on;
fetches PCRE2 when it is not installed), `RP_RUN_MODEL_TESTS` (off; puts the weight-sensitive
GuessIt-corpus test into the default ctest set, which needs a sibling `../guessit` checkout). The
`coverage` preset builds instrumented for gcovr.

## Performance

Measured on an AMD Ryzen 7 5700X3D (8 cores / 16 threads), Release build, 5,000 real names from a
torrent index. All cells come from one process with the cases interleaved: this machine's boost
clock swings run-to-run results by ±25%, so numbers from separate runs are not comparable.
`parse_benchmark` defaults to four batch workers, the number a typical deployment grants a
background scan.

| Path | native C++ | via Python (flat fields) |
|---|---:|---:|
| single-name loop, one thread | 1,547 us/name | — |
| batch, 1 worker | 1,880 us/name | — |
| batch, 4 workers | 714 us/name | ~760 us/name |
| batch, 8 workers | 399 us/name | ~445 us/name |

The Python column includes converting each result into a `ParsedRelease`; the evidence spans cost
about 6% of a batch and `to_dict()` about 8 us per name (`tools/bench_python.py`). With one worker the batch short-circuits to a
plain loop. Bucket sizes 2 through 128 measure identically within noise, only 512 is clearly worse
(12% padding); the default is 32.

Where the time goes, per the stage counters: matmul 64%, convolution 17%, attention 13%, encoding
under 1%. The AVX2 int8 matmul kernel is ALU-bound; VNNI and AVX-512 machines run better kernels.

For provider replies, directory scans, and other grouped inputs, pass the whole collection to one
`BatchParser::parse` call instead of looping over the single-name API.

```sh
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DRP_BUILD_BENCHMARKS=ON
cmake --build build/bench --parallel
build/bench/parse_benchmark models names.txt 3      # one name per line; reports single, batch, padding
```

## Repository layout

```text
include/neurelease/     public C++ API and stable C ABI
src/model/                 preprocessing, model loading, inference, SIMD kernels
src/convert/               typed span-to-value conversion
src/facade/                single and batch parser orchestration
src/abi/                   exception-safe C ABI implementation
model/                    versioned inference artifacts
data/aliases/           editable alias data
bindings/python/           ctypes binding over the C ABI
tools/                     the vocabulary compiler
tests/                     behavior, model, mapping, ABI, robustness, GuessIt-corpus, benchmark
third_party/doctest/       the single vendored test framework
docs/                      result reference, this file, C ABI and comparison documents
```

`build/` and `dist/` appear locally only and are gitignored.
