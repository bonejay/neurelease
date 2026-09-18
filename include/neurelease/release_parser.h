/* neurelease — the C ABI. THE stable contract: everything else in this repository may change
 * shape between versions; this header may only grow.
 *
 * Rules this file lives by, each one enforced by review rather than tooling:
 *   - C99. No C++ type, no exception, no framework object crosses this wall.
 *   - UTF-8 in, UTF-8 out. Span offsets are BYTE offsets into the name as given.
 *   - Every enum value is explicit and append-only. Renumbering is an ABI break; string-keyed
 *     answers are how a "movie" once got read as a special, and neither mistake is repeatable here.
 *   - Ownership is stated per function. A result owns its strings; a batch owns its results;
 *     borrowed pointers say so.
 *   - Errors are rp_status returns plus rp_last_error_message(); nothing throws, nothing aborts.
 *   - rp_abi_version() is checked by every binding at load. It bumps when and only when an
 *     existing symbol or layout changes meaning.
 *
 * Threading: one rp_parser is one-thread-at-a-time. Make one per thread, or use rp_parse_batch,
 * which owns its internal parallelism (physical cores). Results are immutable after creation and
 * may be read from any thread.
 */

#ifndef NEURELEASE_NEURELEASE_H
#define NEURELEASE_NEURELEASE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(RP_BUILDING_SHARED)
#define RP_API __declspec(dllexport)
#elif defined(_WIN32) && defined(RP_USING_SHARED)
#define RP_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define RP_API __attribute__((visibility("default")))
#else
#define RP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped only on breaking change to an existing symbol. Additions do not bump it.
 * 4: rp_result_view grew release_version and release_real. Appending to a struct the CALLER
 * allocates is not a mere addition - a binding compiled against ABI 3 hands rp_view a buffer two
 * int32_t short of what it now writes - so the count moves even though no existing field did. */
#define RP_ABI_VERSION 4u
RP_API uint32_t rp_abi_version(void);

typedef enum rp_status {
    RP_OK = 0,
    RP_ERROR_INVALID_ARGUMENT = 1,
    RP_ERROR_MODEL_NOT_FOUND = 2,
    RP_ERROR_MODEL_INVALID = 3,
    RP_ERROR_OUT_OF_MEMORY = 4,
    RP_ERROR_INTERNAL = 5
} rp_status;

/* Display labels and open values. Closed labels such as source and codec are derived from the typed
 * getters below; open sets such as title, group, and platform carry what the name said. */
typedef enum rp_field {
    RP_FIELD_TITLE = 0,
    RP_FIELD_SCREEN_SIZE = 1,        /* "2160p", "1080p", ... */
    RP_FIELD_SOURCE = 2,         /* "WEB-DL", "BluRay", "CAM", ... */
    RP_FIELD_STREAMING_SERVICE = 3,       /* "AMZN", "NF", "B-GLOBAL", ... */
    RP_FIELD_VIDEO_CODEC = 4,          /* "HEVC", "H.264", "AV1", ... */
    RP_FIELD_AUDIO_CODEC = 5,    /* "DDP", "FLAC", ... */
    RP_FIELD_HDR = 6,            /* the summary; the full set is rp_hdr_formats */
    RP_FIELD_EDITION = 7,        /* the primary; the full list via rp_edition_at */
    RP_FIELD_RELEASE_GROUP = 8,  /* the primary; the full list via rp_release_group_at */
    RP_FIELD_CONTAINER = 9,      /* "MKV", "MP4", ... */
    RP_FIELD_MEDIUM = 10,        /* "video", "audio", "book", ... */
    RP_FIELD_SUBTITLE_FORMAT = 11,
    RP_FIELD_AUDIO_CHANNELS = 12,  /* "5.1", "7.1", "2.0" */
    RP_FIELD_AUDIO_PROFILE = 13    /* "Atmos" */
} rp_field;

typedef enum rp_int_field {
    RP_INT_YEAR = 0,
    RP_INT_SEASON = 1,
    RP_INT_SEASON_END = 2,
    RP_INT_EPISODE = 3,
    RP_INT_EPISODE_END = 4,
    RP_INT_ABSOLUTE_EPISODE = 5,
    RP_INT_EPISODE_COUNT = 6,
    RP_INT_TOKENS = 7, /* how many tokens the model pooled the name to; explains attention cost */
    RP_INT_ABSOLUTE_EPISODE_END = 8
} rp_int_field;

typedef enum rp_flag_field {
    RP_FLAG_VALID = 0,   /* the parse produced a usable candidate */
    RP_FLAG_PACK = 1,
    RP_FLAG_SPECIALS = 2,
    RP_FLAG_COMPLETE_RANGE = 3,
    RP_FLAG_REMUX = 4,
    RP_FLAG_PROPER = 5,
    RP_FLAG_REPACK = 6,
    RP_FLAG_TEN_BIT = 7,
    RP_FLAG_DUAL_AUDIO = 8,
    RP_FLAG_MULTI_AUDIO = 9,
    RP_FLAG_MULTI_SUBS = 10,
    RP_FLAG_HARD_SUBS = 11,
    RP_FLAG_ENGLISH_DUB = 12,
    RP_FLAG_LIGHT_ENCODE = 13,
    RP_FLAG_AI_UPSCALE = 14,
    RP_FLAG_DEGRADED = 15, /* the model could not read this name; only title/year heuristics ran */
    /* Anime as a release convention rather than a nationality: Japanese, Chinese and Korean
     * animation, which the anime databases catalogue together and which are written and numbered
     * alike. A model without the field leaves this clear, so read RP_VERDICT_ANIME's confidence
     * to tell "not anime" from "not asked". */
    RP_FLAG_ANIME = 16
} rp_flag_field;

/* HDR as the bitmask it really is: a DV release routinely carries an HDR10 base layer, and one
 * summary string cannot say so. */
typedef enum rp_hdr_format {
    RP_HDR_HDR10 = 1u << 0,
    RP_HDR_DOLBY_VISION = 1u << 1,
    RP_HDR_HDR10_PLUS = 1u << 2,
    RP_HDR_HLG = 1u << 3
} rp_hdr_format;

/* The whole-name verdicts, typed. Explicit values, append-only. */
typedef enum rp_content_kind {
    RP_CONTENT_UNKNOWN = 0,
    RP_CONTENT_LIVE_ACTION_MOVIE = 1,
    RP_CONTENT_LIVE_ACTION_SERIES = 2,
    RP_CONTENT_ANIMATED_MOVIE = 3,
    RP_CONTENT_ANIMATED_SERIES = 4,
    RP_CONTENT_MUSIC = 5,
    RP_CONTENT_BOOK_DOCUMENT = 6,
    RP_CONTENT_COMIC_MANGA = 7,
    RP_CONTENT_SOFTWARE = 8,
    RP_CONTENT_GAME = 9,
    RP_CONTENT_OTHER = 10,
    /* APPENDED FOR MODEL 3, WHICH NO LONGER GUESSES AT THE MEDIUM. A release name does not say
     * whether a work is animated - nothing in `Shrek.2001.1080p.BluRay.x265` does - so the model
     * states the form and answers the tradition in RP_FLAG_ANIME. The four live-action and
     * animated values above are what a model 2 file returns and are still resolved. */
    RP_CONTENT_MOVIE = 11,
    RP_CONTENT_SERIES = 12
} rp_content_kind;

typedef enum rp_numbering_kind {
    RP_NUMBERING_UNKNOWN = 0,
    RP_NUMBERING_SEASON_EPISODE = 1,
    RP_NUMBERING_ABSOLUTE = 2,
    RP_NUMBERING_VOLUME = 3,
    RP_NUMBERING_DATE = 4,
    RP_NUMBERING_NONE = 5
} rp_numbering_kind;

typedef enum rp_pack_scope {
    RP_PACK_UNKNOWN = 0,
    RP_PACK_SINGLE = 1,
    RP_PACK_EPISODE_BATCH = 2,
    RP_PACK_VOLUME = 3,
    RP_PACK_SEASON = 4,
    RP_PACK_MULTI_SEASON = 5,
    RP_PACK_COMPLETE = 6
} rp_pack_scope;

/* NOTE: "movie" means THE MOVIE ENTRY — the theatrical film as opposed to an episode — and is not
 * a special of any kind. Reading it as one once marked every film in a library a special. */
typedef enum rp_special_kind {
    RP_SPECIAL_UNKNOWN = 0,
    RP_SPECIAL_NONE = 1,
    RP_SPECIAL_OVA = 2,
    RP_SPECIAL_SPECIAL = 3,
    RP_SPECIAL_MOVIE = 4
} rp_special_kind;

typedef enum rp_adult_kind {
    RP_ADULT_UNKNOWN = 0,
    RP_ADULT_NO = 1,
    RP_ADULT_YES = 2
} rp_adult_kind;

typedef enum rp_resolution_tier {
    RP_RESOLUTION_UNKNOWN = 0, RP_RESOLUTION_480P = 1, RP_RESOLUTION_720P = 2,
    RP_RESOLUTION_1080P = 3, RP_RESOLUTION_1440P = 4, RP_RESOLUTION_2160P = 5,
    RP_RESOLUTION_4320P = 6
} rp_resolution_tier;

typedef enum rp_source_kind {
    RP_SOURCE_UNKNOWN = 0, RP_SOURCE_BLURAY = 1, RP_SOURCE_WEB_DL = 2,
    RP_SOURCE_WEB_RIP = 3, RP_SOURCE_HDTV = 4, RP_SOURCE_DVD = 5, RP_SOURCE_CAM = 6,
    RP_SOURCE_SCREENER = 7
} rp_source_kind;

typedef enum rp_video_codec {
    RP_CODEC_UNKNOWN = 0, RP_CODEC_AV1 = 1, RP_CODEC_HEVC = 2, RP_CODEC_H264 = 3,
    RP_CODEC_XVID = 4, RP_CODEC_MPEG2 = 5, RP_CODEC_VP9 = 6, RP_CODEC_VC1 = 7,
    RP_CODEC_WMV = 8, RP_CODEC_VVC = 9, RP_CODEC_VP8 = 10
} rp_video_codec;

typedef enum rp_medium_kind {
    RP_MEDIUM_UNKNOWN = 0, RP_MEDIUM_VIDEO = 1, RP_MEDIUM_MUSIC = 2,
    RP_MEDIUM_AUDIOBOOK = 3, RP_MEDIUM_BOOK = 4, RP_MEDIUM_COMIC = 5,
    RP_MEDIUM_SOFTWARE = 6, RP_MEDIUM_GAME = 7, RP_MEDIUM_IMAGE = 8,
    RP_MEDIUM_SUBTITLE = 9, RP_MEDIUM_ARCHIVE = 10
} rp_medium_kind;

typedef enum rp_subtitle_format {
    RP_SUBTITLE_FORMAT_UNKNOWN = 0, RP_SUBTITLE_FORMAT_SRT = 1,
    RP_SUBTITLE_FORMAT_ASS = 2, RP_SUBTITLE_FORMAT_VOBSUB = 3,
    RP_SUBTITLE_FORMAT_PGS = 4, RP_SUBTITLE_FORMAT_VTT = 5,
    RP_SUBTITLE_FORMAT_SMI = 6
} rp_subtitle_format;

typedef enum rp_edition_kind {
    RP_EDITION_UNKNOWN = 0, RP_EDITION_IMAX = 1, RP_EDITION_CRITERION = 2,
    RP_EDITION_OPEN_MATTE = 3, RP_EDITION_REMASTERED = 4, RP_EDITION_UNRATED = 5,
    RP_EDITION_UNCUT = 6, RP_EDITION_UNCENSORED = 7, RP_EDITION_SPECIAL = 8,
    RP_EDITION_DELUXE = 9, RP_EDITION_REDUX = 10, RP_EDITION_EXTENDED = 11,
    RP_EDITION_DIRECTORS_CUT = 12, RP_EDITION_FINAL_CUT = 13,
    RP_EDITION_THEATRICAL = 14,
    /* Append-only, and the numbers are the C++ EditionKind's: the facade casts between them. */
    RP_EDITION_DESPECIALIZED = 15, RP_EDITION_ASSEMBLY_CUT = 16, RP_EDITION_ANNIVERSARY = 17,
    RP_EDITION_SIGNATURE = 18, RP_EDITION_IMPERIAL = 19, RP_EDITION_DIAMOND = 20,
    RP_EDITION_TWO_IN_ONE = 21, RP_EDITION_PREAIR = 22,
    RP_EDITION_INTERNAL = 23, RP_EDITION_LIMITED = 24,
    RP_EDITION_UNTOUCHED = 25, RP_EDITION_DIRFIX = 26, RP_EDITION_CUSTOM = 27,
    RP_EDITION_WIDESCREEN = 28,
    RP_EDITION_DOWNLOAD = 29, RP_EDITION_RETAIL = 30, RP_EDITION_COLLECTOR = 31,
    RP_EDITION_FINAL = 32, RP_EDITION_ORIGINAL = 33, RP_EDITION_FIX = 34,
    RP_EDITION_COMPLETE_EDITION = 35, RP_EDITION_UNABRIDGED = 36, RP_EDITION_REENCODE = 37,
    RP_EDITION_NUMBERED = 38, RP_EDITION_REGIONAL = 39, RP_EDITION_HIGH_QUALITY = 40,
    RP_EDITION_ULTIMATE = 41, RP_EDITION_CENSORED = 42, RP_EDITION_FAN_EDIT = 43,
    RP_EDITION_BOOTLEG = 44, RP_EDITION_UNOFFICIAL = 45, RP_EDITION_BONUS = 46,
    RP_EDITION_FESTIVAL = 47, RP_EDITION_MULTI_DISC = 48
} rp_edition_kind;

typedef enum rp_verdict {
    RP_VERDICT_CONTENT = 0,
    RP_VERDICT_ADULT = 1,
    RP_VERDICT_NUMBERING = 2,
    RP_VERDICT_PACK = 3,
    RP_VERDICT_SPECIAL = 4,
    RP_VERDICT_ANIME = 5
} rp_verdict;

typedef enum rp_accuracy {
    RP_ACCURACY_EXACT = 0, /* bit-identical to the scalar reference on every CPU */
    RP_ACCURACY_FAST = 1   /* faster on supported AVX2 CPUs; may alter borderline classifications */
} rp_accuracy;

/* Every evidence field. This is separate from rp_field: origins also include numeric fields,
 * flags, and whole-name verdicts. Explicit and append-only. */
typedef enum rp_origin_field {
    RP_ORIGIN_TITLE = 0, RP_ORIGIN_SUBTITLE = 1, RP_ORIGIN_ALTERNATE_TITLE = 2,
    RP_ORIGIN_EPISODE_TITLE = 3, RP_ORIGIN_YEAR = 4, RP_ORIGIN_AIR_DATE = 5,
    RP_ORIGIN_SEASON = 6, RP_ORIGIN_EPISODE = 7, RP_ORIGIN_ABSOLUTE_EPISODE = 8,
    RP_ORIGIN_PACK_MARKER = 9, RP_ORIGIN_SPECIAL_MARKER = 10,
    RP_ORIGIN_QUALITY = 11, RP_ORIGIN_SOURCE = 12, RP_ORIGIN_PLATFORM = 13,
    RP_ORIGIN_EDITION = 14, RP_ORIGIN_CODEC = 15, RP_ORIGIN_HDR = 16,
    RP_ORIGIN_BIT_DEPTH = 17, RP_ORIGIN_REMUX = 18, RP_ORIGIN_PROPER = 19,
    RP_ORIGIN_REPACK = 20, RP_ORIGIN_AI_UPSCALE = 21, RP_ORIGIN_HYBRID = 22,
    RP_ORIGIN_LIGHT_ENCODE = 23, RP_ORIGIN_THREE_D = 24, RP_ORIGIN_DOWNSCALED = 25,
    RP_ORIGIN_AUDIO = 26, RP_ORIGIN_AUDIO_LANGUAGE = 27, RP_ORIGIN_SUBTITLE_LANGUAGE = 28,
    RP_ORIGIN_DUAL_AUDIO = 29, RP_ORIGIN_MULTI_AUDIO = 30, RP_ORIGIN_MULTI_SUBS = 31,
    RP_ORIGIN_ENGLISH_DUB = 32, RP_ORIGIN_HARD_SUBS = 33, RP_ORIGIN_SUBTITLE_FORMAT = 34,
    RP_ORIGIN_GROUP = 35, RP_ORIGIN_SITE_BANNER = 36, RP_ORIGIN_TRACKER_TAG = 37,
    RP_ORIGIN_CONTAINER = 38, RP_ORIGIN_CRC32 = 39, RP_ORIGIN_MEDIUM = 40,
    RP_ORIGIN_CONTENT_KIND = 41, RP_ORIGIN_ADULT_KIND = 42,
    RP_ORIGIN_NUMBERING_KIND = 43, RP_ORIGIN_PACK_SCOPE = 44,
    RP_ORIGIN_SPECIAL_KIND = 45,
    /* Appended 2026-08-31: decorative franchise branding before the canonical title
       ("007 James Bond"). Additive — existing values are unchanged, and a caller that
       does not know it sees an origin field it can ignore. */
    RP_ORIGIN_FRANCHISE_PREFIX = 46
} rp_origin_field;

/* One located field: what was read, from which bytes, and how sure the model was. */
typedef struct rp_origin_view {
    rp_origin_field field;
    const char* value;      /* canonical value; empty when located but not understood */
    const char* text;       /* the raw bytes the value was read from, verbatim */
    int32_t begin;          /* BYTE offsets into the name; -1 when there is no single span */
    int32_t end;
    float confidence;       /* the model's own probability; 1.0 for derived values */
    int unconverted;        /* located, but the conversion layer had no value for the text */
} rp_origin_view;

typedef struct rp_date {
    int16_t year;  /* 0 when the name stated no date */
    int8_t month;
    int8_t day;
} rp_date;

typedef struct rp_parser rp_parser;
typedef struct rp_result rp_result;
typedef struct rp_batch rp_batch;

/* --- lifecycle ---------------------------------------------------------------------------------- */

/* model_dir_utf8 must point at the model files (segmenter.bin and companions). There is no
 * parser without them: this library ships the learned engine only, plus a degraded title/year
 * fallback for names the model cannot encode (flagged RP_FLAG_DEGRADED). */
RP_API rp_status rp_parser_new(const char* model_dir_utf8, rp_parser** out);
RP_API void rp_parser_free(rp_parser* parser);

/* The last error's human-readable detail, owned by the parser, valid until its next call.
 * rp_parser_new failures report through the global variant. */
RP_API const char* rp_last_error_message(const rp_parser* parser);
RP_API const char* rp_global_error_message(void);

/* Process-wide dispatch policy. Set during startup, before constructing or using parsers. */
RP_API rp_status rp_set_accuracy(rp_accuracy accuracy);
RP_API rp_accuracy rp_get_accuracy(void);

/* --- parsing ------------------------------------------------------------------------------------ */

RP_API rp_status rp_parse(rp_parser* parser, const char* name_utf8, rp_result** out);

/* The whole reply at once: spreads the names over the physical cores. Results come back in input
 * order; rp_batch_at borrows from the batch and is valid until rp_batch_free. */
RP_API rp_status rp_parse_batch(rp_parser* parser, const char* const* names_utf8, size_t count,
                                rp_batch** out);

/* How many worker threads rp_parse_batch may use. 0 restores the default (half the logical
 * cores). Takes effect when the batch engine is (re)built: call it before the first
 * rp_parse_batch, or to rebuild with a new count afterwards. */
RP_API rp_status rp_set_batch_threads(rp_parser* parser, int threads);
RP_API size_t rp_batch_size(const rp_batch* batch);
RP_API const rp_result* rp_batch_at(const rp_batch* batch, size_t index);
RP_API void rp_batch_free(rp_batch* batch);

/* --- reading a result ---------------------------------------------------------------------------- */

/* Strings are UTF-8, owned by the result, valid until rp_result_free; NULL means "the name never
 * said". Integers are 0 when unset; flags are 0/1. */
RP_API const char* rp_str(const rp_result* result, rp_field field);
RP_API int32_t rp_int(const rp_result* result, rp_int_field field);
RP_API int rp_flag(const rp_result* result, rp_flag_field field);
RP_API uint32_t rp_hdr_formats(const rp_result* result); /* rp_hdr_format bits */
RP_API rp_date rp_date_value(const rp_result* result);

RP_API rp_content_kind rp_content(const rp_result* result);
RP_API rp_numbering_kind rp_numbering(const rp_result* result);
RP_API rp_pack_scope rp_pack(const rp_result* result);
RP_API rp_special_kind rp_special(const rp_result* result);
RP_API rp_adult_kind rp_adult(const rp_result* result);
/* Whether the release is anime. Zero both for a release that is not and for a model that was never
 * asked; rp_verdict_confidence(result, RP_VERDICT_ANIME) is zero only in the second case. */
RP_API int rp_anime(const rp_result* result);
RP_API rp_resolution_tier rp_screen_size(const rp_result* result);
RP_API rp_source_kind rp_source(const rp_result* result);
RP_API rp_video_codec rp_video_codec_value(const rp_result* result);
RP_API rp_medium_kind rp_medium(const rp_result* result);
RP_API rp_subtitle_format rp_subtitle_format_value(const rp_result* result);
RP_API float rp_verdict_confidence(const rp_result* result, rp_verdict which);

/* List counts make iteration explicit in bindings. Indexed values borrow from the result;
 * index past the end returns the enum's UNKNOWN value or NULL. */
RP_API size_t rp_edition_count(const rp_result* result);
RP_API rp_edition_kind rp_edition_at(const rp_result* result, size_t index);
RP_API size_t rp_release_group_count(const rp_result* result);
RP_API const char* rp_release_group_at(const rp_result* result, size_t index);
RP_API size_t rp_alternative_title_count(const rp_result* result);
RP_API const char* rp_alternative_title_at(const rp_result* result, size_t index);
RP_API size_t rp_language_count(const rp_result* result);
RP_API const char* rp_language_at(const rp_result* result, size_t index);
RP_API size_t rp_subtitle_language_count(const rp_result* result);
RP_API const char* rp_subtitle_language_at(const rp_result* result, size_t index);

/* The evidence: every located field, iterable. The views borrow from the result. */
RP_API size_t rp_origin_count(const rp_result* result);
RP_API rp_status rp_origin_at(const rp_result* result, size_t index, rp_origin_view* out);

/* Everything scalar about a result in ONE call, for bindings whose per-call overhead dwarfs the
 * work (ctypes pays microseconds per crossing; thirty crossings per result was most of the
 * conversion time). Strings are borrowed from the result, valid until rp_result_free; list
 * counts are included so a binding loops only for the lists themselves. Laid out afresh for
 * ABI 3. GROWING IT COSTS AN ABI BUMP - the caller allocates it, so an older binding's buffer is
 * simply too small - which is why release_version/release_real arrived with ABI 4 rather than
 * quietly at the end. */
typedef struct rp_result_view {
    const char* title;
    const char* episode_title;
    const char* franchise_prefix;
    const char* streaming_service;
    const char* audio_codec;
    const char* audio_channels;
    const char* audio_profile;
    const char* hdr;
    const char* container;
    int32_t year, season, season_end, episode, episode_end;
    int32_t absolute_episode, absolute_episode_end, episode_count, tokens;
    /* The revision Sonarr ranks on: the version this file is of its release, and how many times
     * the name said REAL. version is 1 when nothing was stated - the first revision, not a
     * missing one - and a bare PROPER is 2. */
    int32_t release_version, release_real;
    uint8_t screen_size, source, video_codec, medium, content;
    uint8_t numbering, special, adult, pack_scope, subtitle_format;
    uint8_t valid, pack, specials, degraded;
    rp_date date;
    float content_confidence, adult_confidence, numbering_confidence;
    float pack_confidence, special_confidence, anime_confidence;
    uint32_t edition_count, release_group_count, language_count, subtitle_language_count;
    uint32_t alternative_title_count, origin_count;
    /* Every rp_flag_field as one word, bit index = the enum value; and the rp_hdr_format bits.
     * The four flags that already sit above as bytes stay there too. */
    uint32_t flags;
    uint32_t hdr_formats;
    /* Which numeric fields the name actually STATED, because 0 is a value: S00 is a real season
     * and E00 a real episode. Bits: 1 year, 2 season, 4 season_end, 8 episode, 16 episode_end,
     * 32 absolute_episode, 64 absolute_episode_end, 128 episode_count. Derived from the located
     * evidence, so a zero with a span is stated and a zero without one is absent. */
    uint32_t stated;
} rp_result_view;

RP_API rp_status rp_view(const rp_result* result, rp_result_view* out);

/* Copy up to `capacity` origins into `out` in one call; returns how many were written. */
RP_API size_t rp_origins_fill(const rp_result* result, rp_origin_view* out, size_t capacity);

RP_API void rp_result_free(rp_result* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NEURELEASE_NEURELEASE_H */
