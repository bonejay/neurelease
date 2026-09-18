"""Typed Python facade over neurelease's stable C ABI."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field
import re
from enum import Enum, IntEnum
from pathlib import Path
from typing import Iterable


ABI_VERSION = 4


class NeureleaseError(RuntimeError):
    pass


class Kind(str, Enum):
    """A closed value that is also its own label.

    `result.source is SourceKind.BLURAY`, `result.source == "BluRay"` and `str(result.source)`
    all hold, and `json.dumps` writes the label. Each member also carries the C ABI's integer as
    `.id`; the label is what a person reads, the id is what crosses the C boundary.
    """

    def __new__(cls, ident: int, label: str):
        member = str.__new__(cls, label)
        member._value_ = label
        member.id = ident
        return member

    def __str__(self) -> str:
        return str.__str__(self)

    def __format__(self, spec: str) -> str:
        return str.__format__(str(self), spec)

    @classmethod
    def from_id(cls, ident: int):
        table = _ID_TABLES.get(cls)
        if table is None:
            table = _ID_TABLES[cls] = {member.id: member for member in cls}
        return table[ident]


_ID_TABLES: dict = {}


class ContentKind(Kind):
    UNKNOWN = (0, 'unknown')
    # The four values a model 2 file answers. Kept so an older model still resolves; a model 3
    # file never returns them, because a release name does not say whether a work is animated.
    LIVE_ACTION_MOVIE = (1, 'live_action_movie')
    LIVE_ACTION_SERIES = (2, 'live_action_series')
    ANIMATED_MOVIE = (3, 'animated_movie')
    ANIMATED_SERIES = (4, 'animated_series')
    MUSIC = (5, 'music')
    BOOK_DOCUMENT = (6, 'book_document')
    COMIC_MANGA = (7, 'comic_manga')
    SOFTWARE = (8, 'software')
    GAME = (9, 'game')
    OTHER = (10, 'other')
    # Model 3 onward. The four live-action and animated values above are what a model 2 file
    # answers: a release name does not say whether a work is animated, so the form is stated here
    # and the tradition in `anime`.
    MOVIE = (11, 'movie')
    SERIES = (12, 'series')


class ResolutionTier(Kind):
    UNKNOWN = (0, 'unknown')
    P480 = (1, '480p')
    P720 = (2, '720p')
    P1080 = (3, '1080p')
    P1440 = (4, '1440p')
    P2160 = (5, '2160p')
    P4320 = (6, '4320p')


class SourceKind(Kind):
    UNKNOWN = (0, 'unknown')
    BLURAY = (1, 'BluRay')
    WEB_DL = (2, 'WEB-DL')
    WEB_RIP = (3, 'WEBRip')
    HDTV = (4, 'HDTV')
    DVD = (5, 'DVD')
    CAM = (6, 'CAM')
    SCREENER = (7, 'Screener')


class VideoCodec(Kind):
    UNKNOWN = (0, 'unknown')
    AV1 = (1, 'AV1')
    HEVC = (2, 'HEVC')
    H264 = (3, 'H264')
    XVID = (4, 'XviD')
    MPEG2 = (5, 'MPEG2')
    VP9 = (6, 'VP9')
    VC1 = (7, 'VC1')
    WMV = (8, 'WMV')
    VVC = (9, 'VVC')
    VP8 = (10, 'VP8')
    REAL_VIDEO = (11, 'RealVideo')


class MediumKind(Kind):
    UNKNOWN = (0, 'unknown')
    VIDEO = (1, 'video')
    MUSIC = (2, 'music')
    AUDIOBOOK = (3, 'audiobook')
    BOOK = (4, 'book')
    COMIC = (5, 'comic')
    SOFTWARE = (6, 'software')
    GAME = (7, 'game')
    IMAGE = (8, 'image')
    SUBTITLE = (9, 'subtitle')
    ARCHIVE = (10, 'archive')


class SpecialKind(Kind):
    UNKNOWN = (0, 'unknown')
    NONE = (1, 'none')
    OVA = (2, 'OVA')
    SPECIAL = (3, 'special')
    MOVIE = (4, 'movie')


class AdultKind(Kind):
    UNKNOWN = (0, 'unknown')
    NO = (1, 'no')
    YES = (2, 'yes')


class PackScope(Kind):
    UNKNOWN = (0, 'unknown')
    SINGLE = (1, 'single')
    EPISODE_BATCH = (2, 'episode_batch')
    VOLUME = (3, 'volume')
    SEASON = (4, 'season')
    MULTI_SEASON = (5, 'multi_season')
    COMPLETE = (6, 'complete')


class NumberingKind(Kind):
    UNKNOWN = (0, 'unknown')
    SEASON_EPISODE = (1, 'season_episode')
    ABSOLUTE = (2, 'absolute')
    VOLUME = (3, 'volume')
    DATE = (4, 'date')
    NONE = (5, 'none')


class SubtitleFormat(Kind):
    UNKNOWN = (0, 'unknown')
    SRT = (1, 'SRT')
    ASS = (2, 'ASS')
    VOBSUB = (3, 'VobSub')
    PGS = (4, 'PGS')
    VTT = (5, 'VTT')
    SMI = (6, 'SMI')


class EditionKind(Kind):
    UNKNOWN = (0, 'unknown')
    IMAX = (1, 'IMAX')
    CRITERION = (2, 'Criterion')
    OPEN_MATTE = (3, 'Open Matte')
    REMASTERED = (4, 'Remastered')
    UNRATED = (5, 'Unrated')
    UNCUT = (6, 'Uncut')
    UNCENSORED = (7, 'Uncensored')
    SPECIAL = (8, 'Special Edition')
    DELUXE = (9, 'Deluxe')
    REDUX = (10, 'Redux')
    EXTENDED = (11, 'Extended')
    DIRECTORS_CUT = (12, "Director's Cut")
    FINAL_CUT = (13, 'Final Cut')
    THEATRICAL = (14, 'Theatrical')
    DESPECIALIZED = (15, 'Despecialized')
    ASSEMBLY_CUT = (16, 'Assembly Cut')
    ANNIVERSARY = (17, 'Anniversary')
    SIGNATURE = (18, 'Signature')
    IMPERIAL = (19, 'Imperial')
    DIAMOND = (20, 'Diamond')
    TWO_IN_ONE = (21, '2in1')
    PREAIR = (22, 'Preair')
    INTERNAL = (23, 'Internal')
    LIMITED = (24, 'Limited')
    UNTOUCHED = (25, 'Untouched')
    DIRFIX = (26, 'Dirfix')
    CUSTOM = (27, 'Custom')
    WIDESCREEN = (28, 'Widescreen')
    DOWNLOAD = (29, 'Download')
    RETAIL = (30, 'Retail')
    COLLECTOR = (31, 'Collector')
    FINAL = (32, 'Final')
    ORIGINAL = (33, 'Original')
    FIX = (34, 'Fix')
    COMPLETE_EDITION = (35, 'Complete Edition')
    UNABRIDGED = (36, 'Unabridged')
    REENCODE = (37, 'Re-encode')
    NUMBERED = (38, 'Numbered Edition')
    REGIONAL = (39, 'Regional')
    HIGH_QUALITY = (40, 'High Quality')
    ULTIMATE = (41, 'Ultimate')
    CENSORED = (42, 'Censored')
    FAN_EDIT = (43, 'Fan Edit')
    BOOTLEG = (44, 'Bootleg')
    UNOFFICIAL = (45, 'Unofficial')
    BONUS = (46, 'Bonus')
    FESTIVAL = (47, 'Festival')
    MULTI_DISC = (48, 'Multi-Disc')
    ALTERNATE_CUT = (49, 'Alternate Cut')
    SHORTENED = (50, 'Shortened')
    LEAKED = (51, 'Leaked')
    COLORIZED = (52, 'Colorized')
    FULLSCREEN = (53, 'Fullscreen')
    STANDARD = (54, 'Standard')
    CREDITLESS = (55, 'Creditless')
    RE_RECORDED = (56, 'Re-recorded')


class OriginField(IntEnum):
    TITLE = 0
    SUBTITLE = 1
    ALTERNATE_TITLE = 2
    EPISODE_TITLE = 3
    YEAR = 4
    AIR_DATE = 5
    SEASON = 6
    EPISODE = 7
    ABSOLUTE_EPISODE = 8
    PACK_MARKER = 9
    SPECIAL_MARKER = 10
    QUALITY = 11
    SOURCE = 12
    PLATFORM = 13
    EDITION = 14
    CODEC = 15
    HDR = 16
    BIT_DEPTH = 17
    REMUX = 18
    PROPER = 19
    REPACK = 20
    AI_UPSCALE = 21
    HYBRID = 22
    LIGHT_ENCODE = 23
    THREE_D = 24
    DOWNSCALED = 25
    AUDIO = 26
    AUDIO_LANGUAGE = 27
    SUBTITLE_LANGUAGE = 28
    DUAL_AUDIO = 29
    MULTI_AUDIO = 30
    MULTI_SUBS = 31
    ENGLISH_DUB = 32
    HARD_SUBS = 33
    SUBTITLE_FORMAT = 34
    GROUP = 35
    SITE_BANNER = 36
    TRACKER_TAG = 37
    CONTAINER = 38
    CRC32 = 39
    MEDIUM = 40
    CONTENT_KIND = 41
    ADULT_KIND = 42
    NUMBERING_KIND = 43
    PACK_SCOPE = 44
    SPECIAL_KIND = 45
    FRANCHISE_PREFIX = 46


class Text(str):
    """A string field that knows how sure the model was: `r.title == "Show"` and `r.title.confidence`.

    The confidence is the lowest span probability behind the value. Behaves as a plain `str`
    everywhere else (equality, hashing, JSON)."""

    def __new__(cls, value: str, confidence: float = 1.0):
        member = str.__new__(cls, value)
        member.confidence = float(confidence)
        return member


class Number(int):
    """An integer field with `.confidence`; a plain `int` otherwise (`r.season == 3`, arithmetic)."""

    def __new__(cls, value: int, confidence: float = 1.0):
        member = int.__new__(cls, value)
        member.confidence = float(confidence)
        return member


class Value(str):
    """A closed value with `.confidence`: the label as a `str`, plus `.kind` (the enum member),
    `.name` and `.id`. `r.source == SourceKind.BLURAY` and `r.source == "BluRay"` both hold;
    `r.source.kind is SourceKind.BLURAY` is the identity check."""

    def __new__(cls, kind: Kind, confidence: float = 1.0):
        member = str.__new__(cls, str(kind))
        member.kind = kind
        member.confidence = float(confidence)
        return member

    @property
    def name(self) -> str:
        return self.kind.name

    @property
    def id(self) -> int:
        return self.kind.id


class Items(tuple):
    """A tuple field with `.confidence` (the lowest of its items); items are `Text` or `Value`."""

    __slots__ = ()

    def __new__(cls, items, confidence: float | None = None):
        member = tuple.__new__(cls, items)
        return member

    @property
    def confidence(self) -> float:
        return min((item.confidence for item in self), default=1.0)


@dataclass(frozen=True, slots=True)
class Origin:
    field: OriginField
    value: str
    text: str
    begin: int
    end: int
    confidence: float
    unconverted: bool


@dataclass(frozen=True, slots=True)
class AirDate:
    year: int
    month: int
    day: int
    confidence: float = field(default=1.0, compare=False)


@dataclass(frozen=True, slots=True)
class ParsedRelease:
    # Every value below carries `.confidence` (Text, Number, Value, Items): the model's lowest span
    # probability behind it, or the verdict's probability.
    title: Text | None
    streaming_service: Text | None
    audio_codec: Text | None
    audio_channels: Text | None
    audio_profile: Text | None
    hdr: Text | None
    container: Text | None
    # None when the name does not state the value. 0 is a value, not an absence: S00 is a real
    # season and E00 a real episode, so absence needs its own spelling.
    year: Number | None
    season: Number | None
    season_end: Number | None
    episode: Number | None
    episode_end: Number | None
    absolute_episode: Number | None
    absolute_episode_end: Number | None
    episode_count: Number | None
    tokens: int
    # The revision, the way Sonarr counts it, and plain ints rather than Number | None because an
    # unstated revision is not absent: it is the FIRST one. `release_version` is 1 unless the name
    # raised it - a bare PROPER makes it 2, `v2` makes it 2, PROPER beside `v2` makes it 3 - and
    # `release_real` counts the REAL tokens, which mark a re-do of a bad PROPER without advancing
    # the version.
    release_version: int
    release_real: int
    screen_size: Value
    # The screen size as the name wrote it (`1080P`, `720`, `1920x1080`) and, when it stated exact
    # dimensions, those dimensions. Both come from the evidence spans, so they are None when the
    # parse was made with origins=False.
    screen_size_text: Text | None
    frame_size: tuple[int, int] | None
    # The other title facts: the episode's own name, every complete alternate name of the work in
    # name order, and branding written before the title that is not part of it (`Marvels`,
    # `James Bond 007`).
    # Names follow GuessIt wherever the fact is the same (screen_size, streaming_service,
    # release_group, language, edition, alternative_title, date), singular for the tuples as
    # GuessIt has them; only facts GuessIt lacks carry our own names.
    episode_title: Text | None
    alternative_title: Items
    franchise_prefix: Text | None
    source: Value
    video_codec: Value
    medium: Value
    content: Value
    numbering: Value
    special: Value
    adult: Value
    pack_scope: Value
    subtitle_format: Value
    edition: Items
    release_group: Items
    language: Items
    subtitle_language: Items
    valid: bool
    pack: bool
    specials: bool
    degraded: bool
    # The property flags, straight from the name: each is simply present or absent.
    complete_range: bool
    remux: bool
    proper: bool
    repack: bool
    ten_bit: bool
    dual_audio: bool
    multi_audio: bool
    multi_subs: bool
    hard_subs: bool
    english_dub: bool
    light_encode: bool
    ai_upscale: bool
    # Japanese, Chinese or Korean animation: the tradition a release name announces through its
    # group, its numbering and its title, as opposed to the medium, which it does not state.
    anime: bool
    # HDR as the bitmask it really is - a DV release routinely carries an HDR10 base layer.
    hdr10: bool
    dolby_vision: bool
    hdr10_plus: bool
    hlg: bool
    # The broadcast date, for names numbered by date rather than by episode; None when unstated.
    date: AirDate | None
    # The model's own probability for each whole-name verdict, so a caller can require
    # confidence before acting on content/numbering/pack_scope/special/adult.
    content_confidence: float
    numbering_confidence: float
    pack_scope_confidence: float
    special_confidence: float
    adult_confidence: float
    anime_confidence: float
    origins: tuple[Origin, ...]

    def to_dict(self, *, origins: bool = False) -> dict:
        """The stated facts as plain values under GuessIt's property names where GuessIt has the
        same fact, and under our own where it does not.

        Ours also: `anime`, true for Japanese, Chinese or Korean animation.

    Shared: `title`, `alternative_title`, `episode_title`, `year`, `date`, `season`, `episode`,
        `absolute_episode`, `episode_count`, `screen_size`, `source`, `video_codec`, `audio_codec`,
        `audio_channels`, `audio_profile`, `color_depth`, `streaming_service`, `edition`, `other`,
        `language`, `subtitle_language`, `release_group`, `container`, `crc32`, `website`, `type`,
        `version`.
        Like GuessIt, a fact with one value is a scalar and with several a list, and a range is the
        list of its numbers. Ours: `franchise_prefix`, `content`, `medium`, `adult`, `numbering`,
        `pack_scope`, `special`, `hdr`, `frame_size`, `real`. Unstated facts are absent, flags live in
        `other`, and `confidence` carries one number per emitted key.
        """
        out: dict = {}
        confidence: dict = {}

        def put(key: str, value, sure: float | None) -> None:
            out[key] = value
            if sure is not None:
                confidence[key] = round(sure, 3)

        def scalar_or_list(items):
            plain = [str(item) for item in items]
            return plain[0] if len(plain) == 1 else plain

        def span(start, end):
            return list(range(int(start), int(end) + 1)) if end is not None and end > start else int(start)

        if self.title is not None:
            put("title", str(self.title), self.title.confidence)
        if self.alternative_title:
            put("alternative_title", scalar_or_list(self.alternative_title), self.alternative_title.confidence)
        if self.episode_title is not None:
            put("episode_title", str(self.episode_title), self.episode_title.confidence)
        if self.franchise_prefix is not None:
            put("franchise_prefix", str(self.franchise_prefix), self.franchise_prefix.confidence)
        if self.year is not None:
            put("year", int(self.year), self.year.confidence)
        if self.date:
            put("date", f"{self.date.year:04d}-{self.date.month:02d}-{self.date.day:02d}",
                self.date.confidence)
        if self.season is not None:
            put("season", span(self.season, self.season_end), self.season.confidence)
        if self.episode is not None:
            put("episode", span(self.episode, self.episode_end), self.episode.confidence)
        if self.absolute_episode is not None:
            put("absolute_episode", span(self.absolute_episode, self.absolute_episode_end),
                self.absolute_episode.confidence)
        if self.episode_count is not None:
            put("episode_count", int(self.episode_count), self.episode_count.confidence)
        if self.screen_size.id:
            put("screen_size", str(self.screen_size), self.screen_size.confidence)
        if self.frame_size:
            out["frame_size"] = list(self.frame_size)
        if self.source.id:
            put("source", str(self.source), self.source.confidence)
        if self.video_codec.id:
            put("video_codec", str(self.video_codec), self.video_codec.confidence)
        for key in ("audio_codec", "audio_channels", "audio_profile"):
            value = getattr(self, key)
            if value is not None:
                put(key, str(value), value.confidence)
        if self.ten_bit:
            out["color_depth"] = "10-bit"
        if self.hdr and self.hdr != "SDR":
            put("hdr", str(self.hdr), self.hdr.confidence)
        if self.streaming_service is not None:
            put("streaming_service", str(self.streaming_service), self.streaming_service.confidence)
        if self.edition:
            put("edition", scalar_or_list(self.edition), self.edition.confidence)
        other = [label for flag, label in (
            (self.proper, "Proper"), (self.repack, "Repack"), (self.remux, "Remux"),
            (self.complete_range, "Complete"), (self.dual_audio, "Dual Audio"),
            (self.multi_audio, "Multi Audio"), (self.multi_subs, "Multi Subs"),
            (self.hard_subs, "Hardcoded Subtitles"), (self.english_dub, "English Dub"),
            (self.hdr10, "HDR10"), (self.dolby_vision, "Dolby Vision"), (self.hdr10_plus, "HDR10+"),
            (self.hlg, "HLG"), (self.light_encode, "Light"), (self.ai_upscale, "AI Upscale"),
        ) if flag]
        if other:
            out["other"] = other[0] if len(other) == 1 else other
        # THE REVISION AS NUMBERS, under GuessIt's name where it has the same fact. `other`
        # already says THAT a release is a proper or a repack; it cannot say which revision, and
        # `Anime.01v2` and `Anime.01v3` are different files. Absent when the name states neither,
        # the way every other unstated fact here is absent - a first release is version 1 and
        # saying so on every parse would be noise.
        if self.release_version > 1:
            out["version"] = self.release_version
        if self.release_real:
            out["real"] = self.release_real
        if self.language:
            put("language", scalar_or_list(self.language), self.language.confidence)
        if self.subtitle_language:
            put("subtitle_language", scalar_or_list(self.subtitle_language),
                self.subtitle_language.confidence)
        if self.release_group:
            put("release_group", scalar_or_list(self.release_group), self.release_group.confidence)
        if self.container:
            put("container", str(self.container), self.container.confidence)
        for origin in self.origins:
            if origin.begin < 0:
                continue
            if origin.field is OriginField.CRC32 and "crc32" not in out:
                put("crc32", origin.value or origin.text, origin.confidence)
            elif origin.field is OriginField.SITE_BANNER and "website" not in out:
                put("website", origin.value or origin.text, origin.confidence)
        kind = str(self.content)
        if kind.endswith("movie"):
            put("type", "movie", self.content.confidence)
        elif kind.endswith("series"):
            put("type", "episode", self.content.confidence)
        put("content", kind, self.content.confidence)
        if self.medium.id:
            put("medium", str(self.medium), self.medium.confidence)
        put("adult", self.adult.kind is AdultKind.YES, self.adult.confidence)
        put("anime", self.anime, self.anime_confidence)
        if self.numbering.id and self.numbering.kind is not NumberingKind.NONE:
            put("numbering", str(self.numbering), self.numbering.confidence)
        if self.pack_scope.id and self.pack_scope.kind is not PackScope.SINGLE:
            put("pack_scope", str(self.pack_scope), self.pack_scope.confidence)
        if self.special.id and self.special.kind is not SpecialKind.NONE:
            put("special", str(self.special), self.special.confidence)
        if self.degraded:
            out["degraded"] = True
        out["confidence"] = confidence
        if origins:
            out["origins"] = [{"field": origin.field.name.lower(), "value": origin.value,
                               "text": origin.text, "begin": origin.begin, "end": origin.end,
                               "confidence": round(origin.confidence, 3),
                               **({"unconverted": True} if origin.unconverted else {})}
                              for origin in self.origins if origin.begin >= 0]
        return out


# Which flat field an evidence span feeds, for the per-field confidence.
_FIELD_OF_ORIGIN = {
    "TITLE": "title", "ALTERNATE_TITLE": "alternative_title", "EPISODE_TITLE": "episode_title",
    "FRANCHISE_PREFIX": "franchise_prefix", "YEAR": "year", "AIR_DATE": "date",
    "SEASON": "season", "EPISODE": "episode", "ABSOLUTE_EPISODE": "absolute_episode",
    "QUALITY": "screen_size", "SOURCE": "source", "PLATFORM": "streaming_service", "EDITION": "edition",
    "CODEC": "video_codec", "HDR": "hdr", "BIT_DEPTH": "ten_bit", "AUDIO": "audio_codec",
    "AUDIO_LANGUAGE": "language", "SUBTITLE_LANGUAGE": "subtitle_language",
    "DUAL_AUDIO": "dual_audio", "MULTI_AUDIO": "multi_audio", "MULTI_SUBS": "multi_subs",
    "HARD_SUBS": "hard_subs", "GROUP": "release_group", "SITE_BANNER": "site_banner",
    "TRACKER_TAG": "tracker_tag", "CONTAINER": "container", "CRC32": "crc32",
    "PACK_MARKER": "pack", "SPECIAL_MARKER": "specials", "REMUX": "remux", "PROPER": "proper",
    "REPACK": "repack",
}

_FRAME_SIZE = re.compile(r"(\d{3,4})\s*[xX×]\s*(\d{3,4})")


class _Parser(ctypes.Structure):
    pass


class _Result(ctypes.Structure):
    pass


class _OriginView(ctypes.Structure):
    _fields_ = [
        ("field", ctypes.c_int),
        ("value", ctypes.c_char_p),
        ("text", ctypes.c_char_p),
        ("begin", ctypes.c_int32),
        ("end", ctypes.c_int32),
        ("confidence", ctypes.c_float),
        ("unconverted", ctypes.c_int),
    ]


class _Date(ctypes.Structure):
    _fields_ = [("year", ctypes.c_int16), ("month", ctypes.c_int8), ("day", ctypes.c_int8)]


class _ResultView(ctypes.Structure):
    _fields_ = [
        ("title", ctypes.c_char_p), ("episode_title", ctypes.c_char_p),
        ("franchise_prefix", ctypes.c_char_p), ("streaming_service", ctypes.c_char_p),
        ("audio_codec", ctypes.c_char_p), ("audio_channels", ctypes.c_char_p),
        ("audio_profile", ctypes.c_char_p), ("hdr", ctypes.c_char_p), ("container", ctypes.c_char_p),
        ("year", ctypes.c_int32), ("season", ctypes.c_int32), ("season_end", ctypes.c_int32),
        ("episode", ctypes.c_int32), ("episode_end", ctypes.c_int32),
        ("absolute_episode", ctypes.c_int32), ("absolute_episode_end", ctypes.c_int32),
        ("episode_count", ctypes.c_int32), ("tokens", ctypes.c_int32),
        ("release_version", ctypes.c_int32), ("release_real", ctypes.c_int32),
        ("screen_size", ctypes.c_uint8), ("source", ctypes.c_uint8), ("video_codec", ctypes.c_uint8),
        ("medium", ctypes.c_uint8), ("content", ctypes.c_uint8), ("numbering", ctypes.c_uint8),
        ("special", ctypes.c_uint8), ("adult", ctypes.c_uint8), ("pack_scope", ctypes.c_uint8),
        ("subtitle_format", ctypes.c_uint8), ("valid", ctypes.c_uint8), ("pack", ctypes.c_uint8),
        ("specials", ctypes.c_uint8), ("degraded", ctypes.c_uint8),
        ("date", _Date),
        ("content_confidence", ctypes.c_float), ("adult_confidence", ctypes.c_float),
        ("numbering_confidence", ctypes.c_float), ("pack_confidence", ctypes.c_float),
        ("special_confidence", ctypes.c_float), ("anime_confidence", ctypes.c_float),
        ("edition_count", ctypes.c_uint32), ("release_group_count", ctypes.c_uint32),
        ("language_count", ctypes.c_uint32), ("subtitle_language_count", ctypes.c_uint32),
        ("alternative_title_count", ctypes.c_uint32), ("origin_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("hdr_formats", ctypes.c_uint32),
        ("stated", ctypes.c_uint32),
    ]


class _Batch(ctypes.Structure):
    pass


ParserPointer = ctypes.POINTER(_Parser)
ResultPointer = ctypes.POINTER(_Result)
BatchPointer = ctypes.POINTER(_Batch)


def _candidate_libraries(explicit: str | os.PathLike[str] | None) -> Iterable[str]:
    if explicit is not None:
        yield os.fspath(explicit)
        return
    if configured := os.environ.get("NEURELEASE_LIBRARY"):
        yield configured
    package = Path(__file__).resolve().parent
    # `libneurelease.dll` is what MinGW builds and what the wheel bundles; `neurelease.dll` is the
    # MSVC spelling. Both are looked for, beside the package and in the wheel's `_native/`.
    for name in ("libneurelease.dll", "neurelease.dll", "libneurelease.so", "libneurelease.dylib"):
        yield str(package / "_native" / name)
        yield str(package / name)
    if discovered := ctypes.util.find_library("neurelease"):
        yield discovered


MODEL_FILES = ("segmenter.bin", "character_map.bin", "chinese_japanese.bin",
               "transliteration_japanese.bin", "transliteration_chinese.bin")


def _candidate_models() -> Iterable[Path]:
    """Where the four model files are, when the caller does not say.

    The models are committed in this repository, so making every caller name a directory was
    ceremony over a path that is almost always the same one. The order is: what the environment
    says, then beside the package (a wheel that bundles them), then the repository's own `model/`
    found by walking up from this file, then the working directory.
    """
    if configured := os.environ.get("NEURELEASE_MODELS"):
        yield Path(configured)
    package = Path(__file__).resolve().parent
    yield package / "model"
    for parent in package.parents:
        yield parent / "model"
    yield Path("model")


def _default_models() -> Path:
    seen: list[str] = []
    for candidate in _candidate_models():
        if all((candidate / name).is_file() for name in MODEL_FILES):
            return candidate
        seen.append(str(candidate))
    listed = "\n".join(seen)
    raise NeureleaseError(
        "could not find the model files; pass a directory, or set NEURELEASE_MODELS.\n"
        f"looked in:\n{listed}")


def _load_library(explicit: str | os.PathLike[str] | None) -> ctypes.CDLL:
    failures: list[str] = []
    for candidate in _candidate_libraries(explicit):
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    detail = "\n".join(failures) if failures else "no library candidates"
    raise NeureleaseError(f"could not load neurelease shared library\n{detail}")


def _text(value: bytes | None) -> str | None:
    return None if value is None else value.decode("utf-8")


def _encode_name(name: str) -> bytes:
    """UTF-8, with surrogateescape so byte-preserving filenames pass through.

    A filename read from a POSIX filesystem can carry bytes that are not UTF-8; Python represents
    them as lone surrogates, and a plain encode() raises on those. The C layer already degrades
    malformed bytes to replacement characters, which is the right answer for a filename - so the
    binding hands the original bytes through rather than crashing before the parser can decide.
    """
    return name.encode("utf-8", "surrogateescape")


class Parser:
    def __init__(self, model_directory: str | os.PathLike[str] | None = None,
                 library: str | os.PathLike[str] | None = None,
                 threads: int | None = None):
        """threads caps parse_batch's workers; the default is half the logical cores."""
        model_directory = _default_models() if model_directory is None else model_directory
        self._library = _load_library(library)
        self._configure_functions()
        version = int(self._library.rp_abi_version())
        if version != ABI_VERSION:
            raise NeureleaseError(f"ABI {version} is incompatible with binding ABI {ABI_VERSION}")
        self._handle = ParserPointer()
        status = self._library.rp_parser_new(
            os.fsencode(os.fspath(model_directory)), ctypes.byref(self._handle))
        if status != 0:
            message = _text(self._library.rp_global_error_message()) or "parser construction failed"
            raise NeureleaseError(message)
        if threads is not None:
            if self._library.rp_set_batch_threads(self._handle, int(threads)) != 0:
                raise NeureleaseError(self._error() or "could not set batch threads")

    def _configure_functions(self) -> None:
        library = self._library
        library.rp_abi_version.restype = ctypes.c_uint32
        library.rp_parser_new.argtypes = [ctypes.c_char_p, ctypes.POINTER(ParserPointer)]
        library.rp_parser_new.restype = ctypes.c_int
        library.rp_parser_free.argtypes = [ParserPointer]
        library.rp_global_error_message.restype = ctypes.c_char_p
        library.rp_last_error_message.argtypes = [ParserPointer]
        library.rp_last_error_message.restype = ctypes.c_char_p
        library.rp_parse.argtypes = [ParserPointer, ctypes.c_char_p,
                                     ctypes.POINTER(ResultPointer)]
        library.rp_parse.restype = ctypes.c_int
        library.rp_date_value.argtypes = [ResultPointer]
        library.rp_date_value.restype = _Date
        library.rp_view.argtypes = [ResultPointer, ctypes.POINTER(_ResultView)]
        library.rp_view.restype = ctypes.c_int
        library.rp_origins_fill.argtypes = [ResultPointer, ctypes.POINTER(_OriginView),
                                            ctypes.c_size_t]
        library.rp_origins_fill.restype = ctypes.c_size_t
        library.rp_verdict_confidence.argtypes = [ResultPointer, ctypes.c_int]
        library.rp_verdict_confidence.restype = ctypes.c_float
        library.rp_set_batch_threads.argtypes = [ParserPointer, ctypes.c_int]
        library.rp_set_batch_threads.restype = ctypes.c_int
        library.rp_parse_batch.argtypes = [ParserPointer, ctypes.POINTER(ctypes.c_char_p),
                                           ctypes.c_size_t, ctypes.POINTER(BatchPointer)]
        library.rp_parse_batch.restype = ctypes.c_int
        library.rp_batch_size.argtypes = [BatchPointer]
        library.rp_batch_size.restype = ctypes.c_size_t
        library.rp_batch_at.argtypes = [BatchPointer, ctypes.c_size_t]
        library.rp_batch_at.restype = ResultPointer
        library.rp_batch_free.argtypes = [BatchPointer]
        library.rp_result_free.argtypes = [ResultPointer]
        library.rp_origin_count.argtypes = [ResultPointer]
        library.rp_origin_count.restype = ctypes.c_size_t
        library.rp_origin_at.argtypes = [ResultPointer, ctypes.c_size_t,
                                         ctypes.POINTER(_OriginView)]
        library.rp_origin_at.restype = ctypes.c_int
        library.rp_str.argtypes = [ResultPointer, ctypes.c_int]
        library.rp_str.restype = ctypes.c_char_p
        library.rp_int.argtypes = [ResultPointer, ctypes.c_int]
        library.rp_int.restype = ctypes.c_int32
        library.rp_flag.argtypes = [ResultPointer, ctypes.c_int]
        library.rp_flag.restype = ctypes.c_int
        for function in ("rp_screen_size", "rp_source", "rp_video_codec_value", "rp_medium", "rp_content",
                         "rp_numbering", "rp_special", "rp_adult", "rp_pack",
                         "rp_subtitle_format_value"):
            getattr(library, function).argtypes = [ResultPointer]
            getattr(library, function).restype = ctypes.c_int
        library.rp_edition_count.argtypes = [ResultPointer]
        library.rp_edition_count.restype = ctypes.c_size_t
        library.rp_edition_at.argtypes = [ResultPointer, ctypes.c_size_t]
        library.rp_edition_at.restype = ctypes.c_int
        for prefix in ("release_group", "language", "subtitle_language", "alternative_title"):
            count = getattr(library, f"rp_{prefix}_count")
            count.argtypes = [ResultPointer]
            count.restype = ctypes.c_size_t
            at = getattr(library, f"rp_{prefix}_at")
            at.argtypes = [ResultPointer, ctypes.c_size_t]
            at.restype = ctypes.c_char_p

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.rp_parser_free(self._handle)
            self._handle = ParserPointer()

    def __enter__(self) -> "Parser":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _error(self) -> NeureleaseError:
        return NeureleaseError(_text(self._library.rp_last_error_message(self._handle)) or
                                  "neurelease call failed")

    def _copy_result(self, result: ResultPointer, want_origins: bool = True) -> ParsedRelease:
        """One rp_view call carries every scalar; only the lists loop.

        The first version read each field through its own ABI accessor - about thirty ctypes
        crossings per result - and those crossings, not the data, were most of the conversion
        time. Enums are resolved through their value maps because Enum.__call__ costs a
        surprising amount in a loop.
        """
        library = self._library
        view = _ResultView()
        if library.rp_view(result, ctypes.byref(view)) != 0:
            raise self._error()

        def text_list(prefix: str, count: int) -> tuple[str, ...]:
            at = getattr(library, f"rp_{prefix}_at")
            return tuple(_text(at(result, index)) or "" for index in range(count))

        date = view.date
        stated = view.stated
        flags = view.flags
        origins: tuple[Origin, ...] = ()
        if want_origins and view.origin_count:
            raw = (_OriginView * view.origin_count)()
            filled = library.rp_origins_fill(result, raw, view.origin_count)
            field_of = OriginField._value2member_map_
            origins = tuple(Origin(field=field_of[entry.field],
                                   value=_text(entry.value) or "",
                                   text=_text(entry.text) or "",
                                   begin=int(entry.begin), end=int(entry.end),
                                   confidence=float(entry.confidence),
                                   unconverted=bool(entry.unconverted))
                            for entry in raw[:filled])

        screen_size_text = None
        frame_size = None
        for origin in origins:
            if origin.field is OriginField.QUALITY:
                screen_size_text = origin.text
                if match := _FRAME_SIZE.search(origin.text):
                    frame_size = (int(match.group(1)), int(match.group(2)))
                break
        # Confidence per field: the lowest span probability among the spans that produced it. A
        # per-item list (groups, languages, alternate titles) keeps one value per item in order.
        span_confidence: dict[str, float] = {}
        item_confidence: dict[str, list[float]] = {}
        for origin in origins:
            if origin.begin < 0:
                continue
            key = _FIELD_OF_ORIGIN.get(origin.field.name, origin.field.name.lower())
            span_confidence[key] = min(span_confidence.get(key, 1.0), origin.confidence)
            item_confidence.setdefault(key, []).append(origin.confidence)

        def sure(key: str) -> float:
            return span_confidence.get(key, 1.0)

        def text(value: str | None, key: str) -> Text | None:
            return None if value is None else Text(value, sure(key))

        def number(value: int | None, key: str) -> Number | None:
            return None if value is None else Number(value, sure(key))

        def items(values, key: str) -> Items:
            per_item = item_confidence.get(key, [])
            return Items(Text(v, per_item[i]) if i < len(per_item) else Text(v, sure(key))
                         for i, v in enumerate(values))
        # Absolute numbering carries a stated episode 0 across the ABI; a person reads that as no
        # season-relative episode, so that is what the field says.
        episode = view.episode if stated & 8 else None
        if episode == 0 and stated & 32 and view.absolute_episode:
            episode = None

        return ParsedRelease(
            title=text(_text(view.title), "title"),
            streaming_service=text(_text(view.streaming_service), "streaming_service"),
            audio_codec=text(_text(view.audio_codec), "audio_codec"),
            audio_channels=text(_text(view.audio_channels), "audio_codec"),
            audio_profile=text(_text(view.audio_profile), "audio_codec"),
            hdr=text(_text(view.hdr), "hdr"),
            container=text(_text(view.container), "container"),
            year=number(view.year if stated & 1 else None, "year"),
            season=number(view.season if stated & 2 else None, "season"),
            season_end=number(view.season_end if stated & 4 else None, "season"),
            episode=number(episode, "episode"),
            episode_end=number(view.episode_end if stated & 16 else None, "episode"),
            absolute_episode=number(view.absolute_episode if stated & 32 else None, "absolute_episode"),
            absolute_episode_end=number(view.absolute_episode_end if stated & 64 else None,
                                        "absolute_episode"),
            episode_count=number(view.episode_count if stated & 128 else None, "episode"),
            tokens=view.tokens,
            release_version=int(view.release_version),
            release_real=int(view.release_real),
            screen_size=Value(ResolutionTier.from_id(view.screen_size), sure("screen_size")),
            screen_size_text=text(screen_size_text, "screen_size"),
            frame_size=frame_size,
            episode_title=text(_text(view.episode_title), "episode_title"),
            alternative_title=items(text_list("alternative_title", view.alternative_title_count),
                                   "alternative_title"),
            franchise_prefix=text(_text(view.franchise_prefix), "franchise_prefix"),
            source=Value(SourceKind.from_id(view.source), sure("source")),
            video_codec=Value(VideoCodec.from_id(view.video_codec), sure("video_codec")),
            medium=Value(MediumKind.from_id(view.medium), float(view.content_confidence)),
            content=Value(ContentKind.from_id(view.content), float(view.content_confidence)),
            numbering=Value(NumberingKind.from_id(view.numbering), float(view.numbering_confidence)),
            special=Value(SpecialKind.from_id(view.special), float(view.special_confidence)),
            adult=Value(AdultKind.from_id(view.adult), float(view.adult_confidence)),
            pack_scope=Value(PackScope.from_id(view.pack_scope), float(view.pack_confidence)),
            subtitle_format=Value(SubtitleFormat.from_id(view.subtitle_format), sure("subtitle_format")),
            edition=Items(Value(EditionKind.from_id(library.rp_edition_at(result, index)), sure("edition"))
                           for index in range(view.edition_count)),
            release_group=items(text_list("release_group", view.release_group_count), "release_group"),
            language=items(text_list("language", view.language_count),
                                  "language"),
            subtitle_language=items(text_list("subtitle_language", view.subtitle_language_count),
                                     "subtitle_language"),
            valid=bool(view.valid),
            pack=bool(view.pack),
            specials=bool(view.specials),
            degraded=bool(view.degraded),
            complete_range=bool(flags & 1 << 3),
            remux=bool(flags & 1 << 4),
            proper=bool(flags & 1 << 5),
            repack=bool(flags & 1 << 6),
            ten_bit=bool(flags & 1 << 7),
            dual_audio=bool(flags & 1 << 8),
            multi_audio=bool(flags & 1 << 9),
            multi_subs=bool(flags & 1 << 10),
            hard_subs=bool(flags & 1 << 11),
            english_dub=bool(flags & 1 << 12),
            light_encode=bool(flags & 1 << 13),
            ai_upscale=bool(flags & 1 << 14),
            anime=bool(flags & 1 << 16),
            hdr10=bool(view.hdr_formats & 1),
            dolby_vision=bool(view.hdr_formats & 2),
            hdr10_plus=bool(view.hdr_formats & 4),
            hlg=bool(view.hdr_formats & 8),
            date=AirDate(year=date.year, month=date.month, day=date.day,
                             confidence=sure("date")) if date.year else None,
            content_confidence=view.content_confidence,
            adult_confidence=view.adult_confidence,
            anime_confidence=view.anime_confidence,
            numbering_confidence=view.numbering_confidence,
            pack_scope_confidence=view.pack_confidence,
            special_confidence=view.special_confidence,
            origins=origins,
        )

    def parse(self, name: str, *, origins: bool = True) -> ParsedRelease:
        """origins=False skips the evidence spans; since the one-call view they cost about 6% on a
        batch, so the default is simply everything."""
        result = ResultPointer()
        if self._library.rp_parse(self._handle, _encode_name(name), ctypes.byref(result)) != 0:
            raise self._error()
        try:
            return self._copy_result(result, want_origins=origins)
        finally:
            self._library.rp_result_free(result)

    def parse_batch(self, names: Iterable[str], *,
                    origins: bool = True) -> list[ParsedRelease]:
        encoded = [_encode_name(name) for name in names]
        values = (ctypes.c_char_p * len(encoded))(*encoded)
        batch = BatchPointer()
        pointer = values if encoded else None
        if self._library.rp_parse_batch(self._handle, pointer, len(encoded),
                                        ctypes.byref(batch)) != 0:
            raise self._error()
        try:
            return [self._copy_result(self._library.rp_batch_at(batch, index),
                                       want_origins=origins)
                    for index in range(self._library.rp_batch_size(batch))]
        finally:
            self._library.rp_batch_free(batch)


__all__ = [
    "ABI_VERSION", "AdultKind", "ContentKind", "EditionKind", "Items", "Kind", "MediumKind",
    "Number", "NumberingKind",
    "Origin", "OriginField", "PackScope", "ParsedRelease", "Parser", "NeureleaseError",
    "ResolutionTier", "SourceKind", "SpecialKind", "SubtitleFormat", "Text", "Value", "VideoCodec",
]
