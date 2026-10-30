# Changelog

Each version's section is also its GitHub release notes: the publish workflow copies the section
whose heading names the tag, and refuses to publish a tag that has none.

## 0.2.1 - 2026-10-30

### Parsing
- `Cap.604` is season 6, episode 4 whichever way the model labelled the span; it used to come
  through as episode 604 beside a stated `Temporada 6`.
- A cross may state a range: `1x01-02` and `1x01-1x02` are season 1, episodes 1-2, where they read
  episode 1 with no season.

### Packaging
- Wheels for Linux on ARM (`manylinux_2_28_aarch64`), for musl-based Linux such as Alpine
  (`musllinux_1_2`, x86-64 and ARM), and for Intel Macs (`macosx_11_0_x86_64`), beside the three
  there were. Every wheel is installed into a clean environment and used before anything publishes.
- `neurelease.__version__`.
- The C++ library builds against Xcode 16's libc++, which has no `std::jthread`.

## 0.2.0 - 2026-09-22

### Model
- **Model version 5**, trained on the corrected corpus (SHA-256 `7857dcdb…2c34`). Against GuessIt
  4.4.0 on 3,344 video validation names: 97.88% macro field F1 and 91.54% exact, against 86.74%
  and 52.96%. Method in [docs/GUESSIT_COMPARISON.md](https://github.com/bonejay/neurelease/blob/main/docs/GUESSIT_COMPARISON.md).

### Fields
- The release revision: `version` (an explicit `v2` or `repack2`, a PROPER or REPACK counting one
  more) and `real`, by Sonarr's rule. Both are absent when the name states neither.
- New source kinds: `Web` for a bare `WEB` that does not say which kind, `HDRip`, `Screener`,
  `DigitalCinema` (DCP). `RealVideo` as a codec.
- Dozens of new edition kinds, among them Final, Restored, FanEdit, Festival, Commentary,
  Creditless, ReRecorded and the Japanese DL/package/retail editions.
- Bare `UHD`, `4K` and `ULTRAHD` state 2160p when nothing else gives a resolution.
- A year range (`1954-1956`) answers its start; years before 1900 can be read.
- Compound CJK language tags (`中日双语`) answer every language they name.
- About 11,000 spans the model located and the tables could not convert now convert: 22 more
  languages, bare language codes, Chinese subtitle phrases, broadcast and disc sources.
- Episode spans led by the word (`Ep 2x03`), crossed with `×`, or keeping their season letter
  (`s8e6`, `T01XE08`) read correctly.

### Compatibility
- **C ABI 4.** `rp_result_view` grows the two revision fields, so a consumer built against ABI 3
  is refused by the version check. The Python wheel carries its own library and is unaffected.
- Enum values are only ever appended; no existing number changed. `isRepack` also matches RERIP.

## 0.1.1 - 2026-09-11

- The description PyPI shows.

## 0.1.0 - 2026-09-11

- First release: the C++ library, the C ABI and the Python binding, with model version 3. Wheels
  for Linux x86-64, macOS on Apple Silicon and Windows x86-64.
