# C ABI contract

`include/neurelease/release_parser.h` is the stable binary boundary. It is C11-compatible and
contains no implementation-library or C++ types.

- All text is UTF-8. Origin offsets are byte offsets into the input supplied to `rp_parse`.
- Every enum value is explicit and append-only.
- `rp_parser`, `rp_result`, and `rp_batch` are opaque.
- `rp_result` owns its strings. `rp_batch` owns the results returned by `rp_batch_at`.
- A parser is used by one thread at a time. `rp_parse_batch` owns its internal workers.
- No exception crosses the boundary; failures return `rp_status` and preserve diagnostic text.
- Bindings check `rp_abi_version()` before constructing a parser.
- The process-wide accuracy mode is selected during startup, before parsing begins. Exact is the
  default and agrees with the scalar reference; Fast is an explicit throughput tradeoff on
  supported AVX2 CPUs and can alter borderline classifications.

Closed fields have typed getters. `rp_str` remains available for display labels and genuinely open
values. This keeps bindings convenient without making behavior depend on string comparison.

## ABI 2 (2026-09-03)

`RP_ABI_VERSION` is 2. `rp_result_view` gained three trailing members, `episode_title`,
`franchise_prefix` and `alternate_title_count`, and the list accessors `rp_alternate_title_count`
/ `rp_alternate_title_at` were added; everything before them is unchanged, append-only as always.
The numeric view members keep their `0` for "not stated" together with `stated`, whose bits are now
exactly the engagement of the C++ `std::optional<int>` fields behind them, so a written `S00` is
stated and an absent season is not.

## ABI 3 (2026-09-04)

`RP_ABI_VERSION` is 3. Names follow GuessIt wherever the fact is the same, matching the Python
result and `to_dict()`: `rp_resolution` → `rp_screen_size`, `rp_codec` → `rp_video_codec_value`,
`rp_air_date` → `rp_date_value` (the type keeps the bare name, as `rp_subtitle_format_value` already does), `rp_group_count/at` → `rp_release_group_count/at`,
`rp_audio_language_count/at` → `rp_language_count/at`, `rp_alternate_title_count/at` →
`rp_alternative_title_count/at`; `RP_FIELD_QUALITY` → `RP_FIELD_SCREEN_SIZE`, `RP_FIELD_PLATFORM` →
`RP_FIELD_STREAMING_SERVICE`, `RP_FIELD_CODEC` → `RP_FIELD_VIDEO_CODEC`, `RP_FIELD_GROUP` →
`RP_FIELD_RELEASE_GROUP`. The composed audio string is gone: `RP_FIELD_AUDIO_CODEC` (`DDP`),
`RP_FIELD_AUDIO_CHANNELS` (`5.1`) and `RP_FIELD_AUDIO_PROFILE` (`Atmos`) replace it.
`rp_result_view` was laid out afresh with the renamed members (`streaming_service`, `audio_codec`,
`audio_channels`, `audio_profile`, `screen_size`, `video_codec`, `date`, `release_group_count`,
`language_count`, `alternative_title_count`) and is append-only from here. Enum numbers are unchanged.
