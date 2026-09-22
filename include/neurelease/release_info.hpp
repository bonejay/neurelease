// The parse result, as a C++ value type. This is what the facade returns and what the C ABI
// (release_parser.h) is a flat view of: same fields, same vocabulary, same byte offsets.
//
// Context-free facts read from ONE release name. Provider identity, swarm data, cache state,
// ranking and requested-title repair belong to the consuming application.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace neurelease {

// Every fact the parser can locate in a name. A CLOSED SET, so it is an enum rather than a
// string — the UI, the logs and the CSV all name a field the same way, through fieldName(),
// instead of each inventing its own spelling.
enum class Field : std::uint8_t {
    // identity
    Title, Subtitle, AlternateTitle, EpisodeTitle, Year, AirDate,
    // coverage
    Season, Episode, AbsoluteEpisode, PackMarker, SpecialMarker,
    // video
    Quality, ReleaseSource, Platform, Edition, Codec, Hdr, BitDepth, Remux,
    Proper, Repack, AiUpscale, Hybrid, LightEncode, ThreeD, Downscaled,
    // audio and subtitles
    Audio, AudioLanguage, SubtitleLanguage, DualAudio, MultiAudio, MultiSubs, EnglishDub,
    HardSubs, SubtitleFormat,
    // publication
    Group, SiteBanner, TrackerTag, Container, Crc32, Medium,
    // WHOLE-NAME VERDICTS. No location: they are decided from the name as a whole rather than
    // read off a token, which is why they are origins with no span rather than fields with one.
    ContentKind, AdultKind, NumberingKind, PackScope, SpecialKind,
    // APPENDED, NEVER INSERTED: the C ABI pins these numbers, so a new field goes at the end.
    // FranchisePrefix is decorative branding in front of the canonical title
    // (`007 James Bond`, `Comedy Central Presents`). The segmenter has always been able to mark
    // it and mapping.cpp silently dropped it, so no caller could see it and no benchmark could
    // score it — which is how a model improvement from 0.48 to 0.64 on that shape registered as
    // no change at all.
    FranchisePrefix,
};

[[nodiscard]] std::string_view fieldName(Field field); // "title", "release_source", ...

// THE WHOLE-NAME VERDICTS, TYPED. Exactly the classes the exported model carries, and every one
// has an `Unknown` — the whole point: a retrain that adds a class arrives here as Unknown rather
// than as a string that silently matches the wrong branch, and a switch over these warns when a
// case is missing. Values are pinned to the C ABI's (release_parser.h) so the facade can cast.
enum class ContentKind : std::uint8_t {
    Unknown = 0, LiveActionMovie = 1, LiveActionSeries = 2, AnimatedMovie = 3,
    AnimatedSeries = 4, Music = 5, BookDocument = 6, ComicManga = 7,
    Software = 8, Game = 9, Other = 10,
    // APPENDED, AND THE FOUR ABOVE THEM ARE NOW HISTORY. A release name does not say whether a
    // work is animated - nothing in `Shrek.2001.1080p.BluRay.x265` does - and the model that was
    // asked anyway answered animated FILMS correctly 38% of the time while averaging 0.75
    // confidence when it was wrong. Model 3 onward states the form, which every name does say,
    // and answers the tradition separately in `anime`: an anime series is `Series` with
    // `anime == true`. A model 2 file still resolves the four older values, so both load.
    Movie = 11, Series = 12,
};
enum class NumberingKind : std::uint8_t {
    Unknown = 0, SeasonEpisode = 1, Absolute = 2, Volume = 3, Date = 4, None = 5,
};
enum class PackScope : std::uint8_t {
    Unknown = 0, Single = 1, EpisodeBatch = 2, Volume = 3, Season = 4,
    MultiSeason = 5, Complete = 6,
};
// NOTE `Movie` means THE MOVIE ENTRY — the theatrical film as opposed to an episode — and is not
// a special of any kind. Reading it as one once marked every film in a library a special.
enum class SpecialKind : std::uint8_t {
    Unknown = 0, None = 1, Ova = 2, Special = 3, Movie = 4,
};
enum class AdultKind : std::uint8_t {
    Unknown = 0, No = 1, Yes = 2,
};
// Anime as a release convention, not a nationality: Japanese, Chinese and Korean animation, which
// the anime databases catalogue together and which are written and numbered alike. `Unknown` is
// what a model without the field answers, and is not the same as `No`.
enum class AnimeKind : std::uint8_t {
    Unknown = 0, No = 1, Yes = 2,
};

enum class ResolutionTier : std::uint8_t {
    Unknown = 0, P480 = 1, P720 = 2, P1080 = 3, P1440 = 4, P2160 = 5, P4320 = 6,
};
enum class SourceKind : std::uint8_t {
    Unknown = 0, BluRay = 1, WebDl = 2, WebRip = 3, Hdtv = 4, Dvd = 5, Cam = 6,
    // A REVIEWER'S COPY. Not a cam - a screener is a clean transfer - and not the disc it came
    // from either, because it carries watermarks and is often cut. Appended: the C ABI pins these.
    Screener = 7,
    // THE CINEMA MASTER. A Digital Cinema Package is the file a projector is fed; a DCPRip is a
    // rip of one. Neither is a disc nor a stream, and both outrank every other source here.
    DigitalCinema = 8,
    // A PHYSICAL PRINT, SCANNED. Fan restorations of 35mm and 16mm prints are their own category:
    // not a disc, not a broadcast, and not the telecine the cam family means by that word - a
    // scene TELECINE is a leak, while `35MM.FilmScan` is someone's own scan of their own reel.
    Film = 9,
    // A STREAM THAT DECLINES TO SAY HOW IT WAS TAKEN. `WEB` on its own is neither claim: a WEB-DL
    // is the stream as served, a WEBRip is re-encoded from it, and a name spelling only `WEB` has
    // stated the source without stating which. It was answered WEBRip, which is the bare-UHD fault
    // - a token naming a family answered as one member of it - and 447 names in the hard slice
    // alone spell it. Appended, so every existing value keeps its number.
    Web = 10,
    // AN HD RIP OF UNSTATED ORIGIN. `HDRip`, `FHDRip` and `UHDRip` say the rip is high definition
    // and nothing about where it came from - historically a re-encode of whatever HD source was
    // to hand, disc or stream. Answering WEBRip was the bare-WEB fault in another coat, wrong on
    // 35 hard-slice names where the gold says `hdrip`. GuessIt answers no source at all for it,
    // which is the other honest option; this one keeps the fact that was stated. Appended.
    HdRip = 11,
};
enum class VideoCodec : std::uint8_t {
    Unknown = 0, Av1 = 1, Hevc = 2, H264 = 3, Xvid = 4, Mpeg2 = 5, Vp9 = 6,
    Vc1 = 7, Wmv = 8, Vvc = 9, Vp8 = 10,
    // RealVideo, which `.rmvb` releases still carry. Written `RV10`, `RV20`, `RV30`, `RV40`; the
    // generation is not carried, for the same reason `Numbered` drops its number.
    RealVideo = 11,
};
enum class MediumKind : std::uint8_t {
    Unknown = 0, Video = 1, Music = 2, Audiobook = 3, Book = 4, Comic = 5,
    Software = 6, Game = 7, Image = 8, Subtitle = 9, Archive = 10,
};
enum class SubtitleFormat : std::uint8_t {
    Unknown = 0, Srt = 1, Ass = 2, VobSub = 3, Pgs = 4, Vtt = 5, Smi = 6,
};
enum class EditionKind : std::uint8_t {
    Unknown = 0, Imax = 1, Criterion = 2, OpenMatte = 3, Remastered = 4,
    Unrated = 5, Uncut = 6, Uncensored = 7, SpecialEdition = 8, Deluxe = 9,
    Redux = 10, Extended = 11, DirectorsCut = 12, FinalCut = 13, Theatrical = 14,
    // APPENDED, NEVER INSERTED: the C ABI (rp_edition_kind) pins these numbers, so a new edition
    // goes at the end and nothing above it moves.
    Despecialized = 15, AssemblyCut = 16, Anniversary = 17, Signature = 18,
    Imperial = 19, Diamond = 20, TwoInOne = 21, Preair = 22,
    // SCENE TAGS THE MODEL ALREADY MARKS AS EDITIONS and the tables had no member for, so
    // the span was located and converted to nothing. Found by mining 232,595 sampled release
    // names for spans that need a value and got none: `iNTERNAL` alone is 2,737 non-adult
    // video names. Appended, so every existing value keeps its number.
    Internal = 23, Limited = 24, Untouched = 25, Dirfix = 26, Custom = 27, Widescreen = 28,
    // A DOUJIN RELEASE STATES HOW IT WAS SOLD, and the corpus says so 2,916 times: `DL版`
    // is the download edition, `パッケージ版` and `セル版` the boxed and retail ones. `RETAIL`
    // is the same fact in the western scene. `Collector` completes the set the edition
    // token table already had a word for.
    Download = 29, Retail = 30, Collector = 31,
    // QUEUE 01 of the unmapped-span triage: the six commonest editions the tables had no member
    // for. `Final` is NOT `FinalCut` and the difference is the whole reason it is here - French
    // releases write `S01E08.FiNAL` to mark the LAST EPISODE of a season, 3,133 times in the
    // corpus, and routing that to the director's final cut would be confidently wrong on every
    // one of them. `Fix` is the corrective-rerelease family (FIX, FIXED, PROOFFIX, SYNCFIX,
    // RARFIX, SAMPLEFIX) beside the DIRFIX the table already had.
    Final = 32, Original = 33, Fix = 34, CompleteEdition = 35, Unabridged = 36, Reencode = 37,
    // QUEUE 01, second pass. `Numbered` carries a book's `2ed`/`3rd Edition` - the NUMBER is not
    // carried, only the fact that one was stated, which is all a consumer can act on without an
    // edition-number field. `Regional` is the same compromise for `美版`, `Kinofassung`'s cousin:
    // a region-specific cut is stated, without saying which region. `HighQuality` groups the
    // Chinese encode editions - 高码版 high bitrate, 60帧率版本 sixty frames, 高清版 HD - which all
    // mean "the better of the two encodes we published".
    Numbered = 38, Regional = 39, HighQuality = 40, Ultimate = 41,
    // QUEUE FILE 02. `Censored` is the stated opposite of `Uncensored` and just as convertible.
    // `FanEdit` is a RE-CUT BY SOMEONE OTHER THAN THE STUDIO - a different work from the release
    // it was made out of, which is why it is not folded into Unofficial. `Bootleg` is an
    // unofficial recording, the word the music scene uses; `Unofficial` is the wider claim, an
    // encode or batch nobody official published. `Bonus` is bonus-disc material, which Sonarr
    // models as season extras. `Festival` is the cut shown at festivals, distinct from the
    // theatrical one. `MultiDisc` says a release spans several discs without saying how many -
    // the same compromise `Numbered` makes, for the same reason: there is no field for the count.
    Censored = 42, FanEdit = 43, Bootleg = 44, Unofficial = 45, Bonus = 46, Festival = 47,
    MultiDisc = 48,
    // QUEUE FILE 03. `AlternateCut` is a different cut that claims no direction - unlike
    // `Extended` and `Shortened`, which say which way. `Leaked` marks a release that escaped
    // before its publisher meant it to. `Colorized` is a colourised print of a black-and-white
    // film, a real edition of a real film. `Fullscreen` is the 4:3 transfer, the counterpart of
    // the `Widescreen` the table already had.
    AlternateCut = 49, Shortened = 50, Leaked = 51, Colorized = 52, Fullscreen = 53,
    // QUEUE FILE 04. `Standard` is the counterpart of `Limited` - `通常版` exists precisely to say
    // this is NOT the first-press edition, and answering nothing loses that. `Creditless` is the
    // opening or ending with no credits over it, a standard anime extra the scene also writes
    // NCOP and NCED. `ReRecorded` is a new performance of an existing work, not a new transfer of
    // it: `Taylor's Version` is the spelling the corpus carries, and the distinction matters more
    // than most here - the two recordings are different masters of different takes.
    Standard = 54, Creditless = 55, ReRecorded = 56,
    // QUEUE FILE 05. `Commentary` is the same film with a different audio track over it.
    // `Explicit` is the unbleeped master of a music release - the opposite claim to the `clean`
    // this triage refused, and unambiguous where that word is not. `Reissue` is a later pressing
    // of the same work, which in music is often a different master and always a different SKU.
    Commentary = 57, Explicit = 58, Reissue = 59,
    // QUEUE FILE 06. `OriginalAspectRatio` is NOT `Widescreen`: OAR means the frame the film was
    // shot in, which for a 1950s television production is 4:3 and for a scope feature is 2.39:1.
    // It says the transfer was not reframed, and that is a different claim from either shape.
    OriginalAspectRatio = 60,
    // A RESTORATION IS NOT A REMASTER. It repairs damage - torn frames, faded dye, missing
    // footage - where a remaster re-derives from materials that were never damaged. Both rulers
    // this parser is measured against hold the distinction, and folding the two cost names on
    // each of them.
    Restored = 61,
};

[[nodiscard]] std::string_view label(ResolutionTier value) noexcept;
[[nodiscard]] std::string_view label(SourceKind value) noexcept;
[[nodiscard]] std::string_view label(VideoCodec value) noexcept;
[[nodiscard]] std::string_view label(MediumKind value) noexcept;
[[nodiscard]] std::string_view label(SubtitleFormat value) noexcept;
[[nodiscard]] std::string_view label(EditionKind value) noexcept;

// A calendar date with no library behind it. year == 0 means "the name stated no date", which is
// the normal case — a dated release states a date INSTEAD of a year.
struct Date {
    bool operator==(const Date&) const = default;

    std::int16_t year = 0;
    std::int8_t month = 0;
    std::int8_t day = 0;

    [[nodiscard]] bool valid() const noexcept { return year != 0; }
};

// ONE FACT, ITS VALUE, AND WHERE IT CAME FROM.
//
// The reason this exists: a parser that returns only values cannot be argued with. "quality =
// 1080p" is either believed or not, while "quality = 1080p, read from bytes 24-29, which say
// 1920x1080" can be checked at a glance.
struct FieldOrigin {
    bool operator==(const FieldOrigin&) const = default;

    Field field{};
    std::string value; // the canonical value, identical to what the flat field carries
    std::string text;  // the raw substring that produced it, verbatim
    // BYTE offsets into rawName, so rawName.substr(begin, end - begin) is `text`. BOTH -1 when
    // the parser cannot say where: a whole-name verdict has no span.
    std::int32_t begin = -1;
    std::int32_t end = -1;
    // The model's own probability for the winning reading; 1.0 for values derived rather than
    // located, so a low-confidence answer is visibly different from a certain one.
    float confidence = 1.0F;
    // FOUND, BUT NOT UNDERSTOOD. The model located the field and the conversion layer had no
    // value for the text sitting there, so `value` is empty while `text` says what was read.
    //
    // This is the third outcome, and keeping it distinct is the whole point. "The name never said"
    // and "the name said something we cannot read" look identical in a struct of empty fields, and
    // they call for opposite responses: the first is normal, the second is a table gap or a model
    // mistake. Nothing is ever guessed to avoid one of these — an unrecognised token stays
    // unconverted rather than being rounded to the nearest value we happen to know.
    bool unconverted = false;

    [[nodiscard]] bool located() const noexcept { return begin >= 0 && end > begin; }
};

struct ReleaseInfo {
    bool operator==(const ReleaseInfo&) const = default;

    std::string rawName;
    // WHAT KIND OF THING THIS IS: "video" | "music" | "audiobook" | "book" | "comic" | "software" |
    // "game" | "image" | "subtitle" | "archive", or empty when nothing in the name says.
    //
    // A general torrent index returns a great deal that is not video, and a caller wants to
    // refuse those rather than parse them badly — a Photoshop release with a version number is
    // not a 2024 movie.
    MediumKind medium = MediumKind::Unknown;
    std::string container; // normalised, lower case, no dot: "mkv", "epub", "iso"
    std::string title;
    // THE OTHER TITLE FACTS, as fields rather than only as origins. The episode's own name; every
    // complete alternate name of the work (`Shingeki no Kyojin` beside `Attack on Titan`), in name
    // order; branding written before the title that is not part of it (`Marvels`, `James Bond
    // 007`). Empty when the name has none.
    std::string episodeTitle;
    std::vector<std::string> alternativeTitles;
    std::string franchisePrefix;
    // NUMBERS ARE OPTIONAL BECAUSE 0 IS A VALUE. `S00` is a real season and `E00` a real episode,
    // so "the name did not say" needs its own spelling: nullopt. A caller that wants the old
    // zero-for-absent reading writes `info.year.value_or(0)`.
    std::optional<int> year;
    // The broadcast DATE, for the releases that number by date instead of by episode: daily shows
    // ("Show.2023.06.01"), and a large share of adult releases.
    Date date;
    ResolutionTier screenSize = ResolutionTier::Unknown;
    SourceKind source = SourceKind::Unknown;
    std::string streamingService;
    bool proper = false;
    bool repack = false;
    // THE REVISION, AS SONARR COUNTS IT (QualityParser.ParseQuality -> Revision), because a caller
    // choosing between two files of the same episode ranks on exactly these two numbers.
    //
    // `revisionVersion` IS 1 WHEN THE NAME STATES NOTHING: an unstated revision is the FIRST one,
    // not a missing one, which is why it defaults to 1 rather than 0. A written number (`v2`,
    // `[v3]`, `repack2`) sets it, and a PROPER or a REPACK then adds one on top - so a bare
    // `PROPER` is 2 and `PROPER` beside `v2` is 3. Reading a bare PROPER as version 1 is what lost
    // every version case the Sonarr and Radarr suites disagreed with us on.
    //
    // `revisionReal` COUNTS REAL SEPARATELY instead of folding it into the version: the scene uses
    // REAL for a re-do of a botched PROPER, not for a new revision, so `REAL.REAL.PROPER` is
    // version 2 with real 2.
    int revisionVersion = 1;
    int revisionReal = 0;
    // THE PRIMARY EDITION, and `editions` for all of them — because a release is routinely
    // several at once. "Uncut Unrated DC" is uncut AND unrated AND a director's cut; it is not a
    // choice between them. `edition` is the highest-precedence member of `editions`; the
    // precedence describes how strongly an edition IDENTIFIES a release, not its quality.
    EditionKind edition = EditionKind::Unknown;
    std::vector<EditionKind> editions;
    VideoCodec videoCodec = VideoCodec::Unknown;
    // THE AUDIO, IN ITS PARTS: the codec (`DDP`, `FLAC`), the channel layout (`5.1`) and the
    // feature profile (`Atmos`), each empty when the name does not say.
    std::string audioCodec;
    std::string audioChannels;
    std::string audioProfile;
    std::string hdr; // the summary; the full set is the four booleans below
    // THE FIRST RELEASE GROUP, AND ALL OF THEM.
    //
    // A release names more than one team as often as not — an encoder in the tail and a publisher
    // in brackets. `releaseGroup` stays the primary one (in name order) because most callers key
    // on a single value; `releaseGroups` is the whole list, primary included. The label schema keeps
    // release_group neutral and repeatable for exactly this reason: it does not claim who did what.
    std::string releaseGroup;
    std::vector<std::string> releaseGroups;

    std::optional<int> season;
    std::optional<int> seasonEnd;
    std::optional<int> episode;
    std::optional<int> episodeEnd;
    std::optional<int> absoluteEpisode;
    std::optional<int> absoluteEpisodeEnd;
    std::optional<int> episodeCount;
    bool pack = false;
    bool specials = false;
    bool explicitCompleteRange = false;

    bool hdr10 = false;
    bool dolbyVision = false;
    bool hdr10Plus = false;
    bool hlg = false;
    bool remux = false;
    bool aiUpscale = false;
    bool tenBit = false;
    bool hybrid = false;
    bool lightEncode = false;
    bool threeD = false;
    bool downscaled = false;
    bool dualAudio = false;
    bool englishDub = false;
    bool multiAudio = false;
    bool multiSubs = false;
    // BURNED INTO THE PICTURE. "HC", "KORSUB", "Hardcoded Eng Subs": subtitles that cannot be
    // turned off, which for a library is a defect rather than a feature — the same release
    // without them is strictly better, and nothing else in this struct can say so.
    bool hardSubs = false;
    // "SRT" | "ASS" | "VOBSUB" | "PGS" | "VTT" | "SMI", or empty when the name does not say.
    // Text formats can be restyled and searched; PGS and VOBSUB are pictures of text.
    SubtitleFormat subtitleFormat = SubtitleFormat::Unknown;
    bool originalAudioIncluded = false;
    std::vector<std::string> languages;
    std::vector<std::string> subtitleLanguages;
    // ANIME, THE DISTINCTION A NAME ACTUALLY MAKES: the bracketed fansub group, the CRC32, the
    // absolute episode, the romaji title. Japanese, Chinese and Korean animation all count, since
    // they are released and numbered alike and the anime databases catalogue them together;
    // Western animation does not. False on a model that predates the field.
    bool anime = false;
    bool valid = false;
    // The model could not read this name (it would not encode); only the title/year heuristics
    // ran, and everything else is empty. Rare, and worth surfacing rather than hiding: a caller
    // ranking candidates should trust a degraded parse less.
    bool degraded = false;

    // THE WHOLE-NAME VERDICTS, TYPED — and these are the ones to branch on, parsed once so no
    // caller has to compare a label. Each with the model's own probability for the winner.
    ContentKind content = ContentKind::Unknown;
    NumberingKind numbering = NumberingKind::Unknown;
    PackScope packScope = PackScope::Unknown;
    SpecialKind special = SpecialKind::Unknown;
    AdultKind adult = AdultKind::Unknown;
    float contentConfidence = 0.0F;
    float animeConfidence = 0.0F;
    float numberingConfidence = 0.0F;
    float packScopeConfidence = 0.0F;
    float specialConfidence = 0.0F;
    float adultConfidence = 0.0F;

    // Everything the model located, in name order. The flat fields above stay the fast path;
    // origins are for the caller that wants to SHOW or CHECK the reading rather than only act
    // on it. NO TIMING HERE, deliberately: two parses of the same name are the same ANSWER even
    // when one took longer, and equality over this struct is how batch-equals-single is checked.
    // The facade reports timings beside the result, never inside it.
    std::vector<FieldOrigin> origins;
};

// The first origin recorded for `field`, or nullptr. Fields that legitimately repeat — a name
// with two groups, several audio languages — keep every occurrence in `origins`; this is the
// convenience for the single-valued majority.
[[nodiscard]] const FieldOrigin* originOf(const ReleaseInfo& info, Field field);

} // namespace neurelease
