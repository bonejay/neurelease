#pragma once

// ============================================================================
// THE CONVERSION LAYER — ONE VOCABULARY.
// ============================================================================
//
// Turning found text into a canonical value is the parser's actual product, and it must not
// depend on WHO found the text. "1920x1080", "1080P" and "FullHD" are one answer, "E-AC-3" and
// "DD+" are one answer, whether the text was a whole name or the one span the model typed.
//
// So every field is one function here, and it answers both questions at once:
//
//     std::string value = convert::resolutionValue(typedSpan);
//
// The learned model already located and typed each span. This layer therefore converts values; it
// does not scan whole names or duplicate the model with a second locator engine.

#include "neurelease/release_info.hpp" // Date

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace neurelease::convert {

// One converted field. `begin`/`end` are BYTE offsets into the typed span passed to the converter;
// the mapper adds the span's source offset when recording evidence against the whole name.
struct Reading {
    std::string value; // the canonical value; empty when the field is not stated here
    std::string text;  // the raw substring that produced it, verbatim
    std::int32_t begin = -1;
    std::int32_t end = -1;

    [[nodiscard]] bool found() const noexcept { return !value.empty(); }
};

// The dynamic-range claims, ordered so a comparison reads naturally: a later value is a stronger
// claim than an earlier one.
enum class HdrFormat : std::uint8_t { Sdr, Hlg, Hdr10, Hdr10Plus, DolbyVision };

[[nodiscard]] std::string_view hdrFormatLabel(HdrFormat format); // "HLG" | "HDR10" | "HDR10+" | "DV"
[[nodiscard]] inline bool hdrFormatIsWanted(HdrFormat format) { return format != HdrFormat::Hlg; }

// --- video ------------------------------------------------------------------------------
//
// HIGHEST TIER WINS inside a typed resolution span. A compound span can contain both the target
// size and a source size, so the table is walked in tier order.
ResolutionTier resolutionValue(std::string_view token);

// "BluRay" | "WEB-DL" | "WEBRip" | "HDTV" | "DVD" | "CAM" | empty.
SourceKind sourceValue(std::string_view token);
// A source token that is NOTHING BUT "UHD" (or "4K"). In the source position that usually means
// the 4K Blu-ray ("Movie.2160p.UHD.x265"), which is why sourceValue answers BluRay for it — but
// the word also rides along in web releases ("2160p.UHD.AMZN.WEB-DL") where it only restates the
// resolution. So the reading is TENTATIVE: a caller that later sees a definitive source span
// should let that one win. This predicate marks the tentative case.
bool sourceTokenIsBareUhd(std::string_view token);
// Whether a source token also states a REMUX. Separate from the source value on purpose: a
// "BDRemux" is a Blu-ray AND a remux, and folding one into the other loses a fact a ranker
// weighs on its own.
bool sourceTokenIsRemux(std::string_view token);
// Whether a source token states a LIGHT ENCODE — "MicroHD", "HDLight", "mHD". A separate question
// for the same reason as remux: it is a fact about the encode, not about where the video came
// from, and the consumer already has a field for it.
bool sourceTokenIsLightEncode(std::string_view token);

// "AV1" | "HEVC" | "H.264" | empty.
VideoCodec codecValue(std::string_view token);

// The dynamic range CLAIMED here, with the precedence (DV, then HDR10+, then HLG, then plain
// HDR) in one place so no caller can label a Dolby Vision release HDR10 by testing in the wrong
// order. `format` is the enum the flags come from; `reading.value` is the display label.
struct HdrReading {
    HdrFormat format = HdrFormat::Sdr;
    Reading reading;
};
// The same question asked of a token the caller already knows is an HDR tag, and therefore asked
// permissively: "UHDR", "4K_HDR" and "DVHDR" are compound typed tags whose HDR word has no word
// boundary in front of it.
HdrReading hdrValue(std::string_view token);
// "10bit" | "8bit" | empty.
Reading bitDepthIn(std::string_view text);

// "Director's Cut" | "Extended" | "IMAX" | ... | empty. Most specific first.
EditionKind editionIn(std::string_view text);

// EVERY EDITION THE TEXT STATES, ordered by precedence — "Uncut.Unrated.DC" gives
// {Unrated, Uncut, Director's Cut}, because the release is all three at once. `editionIn` stays
// the single best answer and is exactly the first element of this list.
//
// The order is a PRECEDENCE over how strongly an edition identifies a release, not a quality
// ranking: the transfer and authority markers (IMAX, Criterion, Remastered) first, because they
// say where the picture came from; then the content markers (Unrated, Uncut, Uncensored); then
// the cut markers (Extended, Director's Cut, Redux, Theatrical, Final Cut), which are the most
// common and the least distinctive — a director's cut of a remaster is still most usefully
// described as the remaster.
std::vector<EditionKind> editionsIn(std::string_view text);

// --- audio ------------------------------------------------------------------------------
//
// Each audio span is typed independently by the model and kept as its own field.
// "DDP" | "DD" | "DTS-HD MA" | "DTS:X" | "TrueHD" | "AAC" | ... | empty.
std::string audioCodecValue(std::string_view token);
// "5.1" | "7.1" | "2.0" | empty. "6CH" is 5.1; a flattened "5 1" is 5.1.
std::string audioChannelsValue(std::string_view token);
// The object-audio format layered over the codec: "Atmos" | "DTS:X" | "Auro-3D" | empty. Written
// every way a release name writes it - `DTS-X`, `DTS.X`, `DTSX`, `Atmos`, `ATMOS` - and separate
// from the codec because a name states both (`DDP 5.1 Atmos`, `DTS-HD MA 7.1 DTS:X`).
std::string audioProfileValue(std::string_view token);

// WHAT COUNTS AS A YEAR. Fixed rather than read from the clock, because a parser whose answers
// change with the date is not reproducible, and a name stating a year a decade out is stating
// something else - `Blade Runner 2049` and `Cyberpunk 2077` are titles, not release years.
// CINEMA IS OLDER THAN THE REGEX ALLOWED. `Movie Name (1897) [DVD].mp4` lost its year twice
// over: the reader matched only `19xx` and `20xx`, so an 1890s film could not be read at all,
// and this bound would have refused it anyway. The Lumiere programmes date from 1895.
inline constexpr int PlausibleYearFirst = 1890;
inline constexpr int PlausibleYearLast = 2035;

// --- coverage markers ---------------------------------------------------------------------
//
// The numbers a season, episode or volume marker states, whatever notation wrote them: "S01" is
// 1, "S01-S04" is 1 through 4, "01-03" is 1 through 3, "Vol. 04" is 4, "16.episodes" is 16. The
// model already decided this text IS a marker; the only question left is which numbers it states.
struct MarkerNumbers {
    int first = 0;
    int last = 0; // 0 unless the marker states a range
    // HOW MANY VALUES THE MARKER ACTUALLY NAMED, which is not last - first + 1 when the marker
    // enumerates with gaps ("Season.1.3.5" names three, spanning five). 1 for a single number.
    int count = 0;
    // ZERO IS A NUMBER HERE. "S00" and "E00" are where specials are filed, so "stated no number"
    // and "stated zero" cannot both be first == 0 — the caller asks this instead.
    bool stated = false;
};
MarkerNumbers markerNumbersIn(std::string_view text);

// WHAT AN EPISODE MARKER'S NOTATION STATES, which is not the same question as which numbers are in
// it. `2x11` is season 2 episode 11 and `4of10` is episode 4 of ten; read as bare numbers by
// markerNumbersIn both become the false ranges 2-11 and 4-10. The model has already decided this
// span IS an episode marker and where it ends; this reads the notation it is written in.
struct EpisodeMarkerReading {
    int season = 0; // 0 unless the notation states one, as `2x11` does
    int first = 0;
    int last = 0;   // 0 unless the marker states a range
    int count = 1;
    bool stated = false;
};
EpisodeMarkerReading episodeMarkerIn(std::string_view text);

// WHAT A COMBINED MARKER'S SINGLE RUN OF DIGITS STATES. `Cap.104` is season 1 episode 4: the
// trailing TWO digits are the episode and everything before them the season. Unlike every other
// notation rule here this one is NOT applied on the strength of the characters - the model has to
// have typed the span `season_episode_marker` first, because `104` and the `102` of an anime at
// episode 102 are the same three digits and only the surrounding convention tells them apart.
// That decision is the model's; this only reads the digits once it has been made.
EpisodeMarkerReading seasonEpisodeMarkerIn(std::string_view text);

// --- container and medium ----------------------------------------------------------------
//
// WHAT KIND OF THING THIS RELEASE IS, from the one piece of evidence that is never ambiguous.
//
// A file extension is a closed set and it does not lie: ".epub" is a book, ".cbz" a comic, ".iso"
// and ".exe" software, ".m4b" an audiobook, ".mkv" video. For a video library that is worth more
// than it looks — a third of what a general torrent index returns is not video at all, and the
// difference between "we parsed this badly" and "this is a Photoshop release" is exactly this.
std::string containerValue(std::string_view token); // "mkv" | "epub" | "iso" | ... | empty
// "video" | "music" | "book" | "comic" | "software" | "game" | "image" | "subtitle" | "archive"
MediumKind mediumOfContainer(std::string_view container);
// An AUDIO sample depth, which is not a video bit depth: "24bit", "16Bit-44.1kHz", "24-96".
bool statesAudioSampleDepth(std::string_view token);
// Software bit-ness: "64bit", "32-bit", "21.1-xfce-64bit".
bool statesSoftwareBitness(std::string_view token);

// --- languages ---------------------------------------------------------------------------
//
// THE CODES A TOKEN STATES, THROUGH THE DECORATION RELEASE NAMES WRAP THEM IN.
//
// Three stages, because a fixed keyword list only ever catches the first one:
//   1. the token as it stands            "SWE", "français", "中文字幕"
//   2. the token with its subtitle words and counts stripped
//                                        "EngSub" -> "Eng", "2xUKR" -> "UKR", "CZtit" -> "CZ"
//   3. the token split into parts, on separators AND on camel-case humps
//                                        "ENSUB+PLSUB" -> eng, pol; "RuEn" -> rus, eng
// Empty when nothing in it names a language — which is the honest answer for "ASS" or "Subbed".
std::vector<std::string> languageCodesOfToken(std::string_view token);

// THE SUBTITLE FORMAT, which is a fact about the file and not about the language.
//
// "SRT" | "ASS" | "VOBSUB" | "PGS" | "VTT" | "SMI", or empty. The distinction that matters is
// text against bitmap: SRT and ASS are text, so a player can restyle them, scale them, or hand
// them to a search index; PGS and VOBSUB are pictures of text, so it can do none of that.
//
// The caller supplies a span already typed as a subtitle field or container.
SubtitleFormat subtitleFormatValue(std::string_view token);

// A token that says subtitles EXIST without naming a language: a container format ("ASS", "SRT",
// "PGS", "SUP", "VOBSUB"), or a generic word ("Subbed", "w.subs", "INC SUBS"). Not a language,
// and not a gap either — the name really did only say that much.
bool statesSubtitlesOnly(std::string_view token);

// ...and a token that says there are SEVERAL: "Multi Subs", "MultiSub", "SRTx2", "Dual Subs".
bool statesSeveralSubtitles(std::string_view token);

// --- identity ---------------------------------------------------------------------------

// The tail of title reading: separators to spaces, a duplicated title collapsed, trailing
// punctuation gone, and a foreign-script prefix dropped when a Latin title survives beside it.
// Runs on the span the model typed as the main title — location was the model's job; this is
// value.
std::string titleText(std::string_view raw);

// The convention preferLatinTitle serves: a local site prepends its translation to the real
// title, and the Latin half is the one worth keeping. Public because titleText applies it LAST,
// and a caller assembling a title from several spans needs to apply it once at the end instead.
std::string preferLatinTitle(std::string_view title);

// A group name out of a group token: brackets and surrounding punctuation gone, a container
// extension stripped ("-GRP.mkv" is GRP) while a domain is kept whole ("MkvCage.ws"). Empty when
// the token is all digits.
std::string groupText(std::string_view raw);
// The tag words that are never a group name, however group-shaped their position is: "x264",
// "AAC", "COMPLETE". Location guards need this too, so it is shared rather than private.
bool isNeverAGroup(std::string_view word);

// THE PLATFORM VOCABULARY: canonical names are a FIXED SET, membership is OPEN.
//
// Built from the label corpus, not from memory: 31,907 platform spans carry 1,321 distinct
// spellings, of which the top 60 cover 84.5% and 40% of the vocabulary appears exactly once
// (1.6% of spans). So a curated table earns almost all of the value and a closed set would
// throw away the tail for nothing — an unrecognised tag is still evidence that a release came
// from somewhere specific.
//
// What the table is FOR is comparability. Upper-casing alone left "B-Global", "BGlobal" and
// "BStation" as three different values for one service, and "NF" and "Netflix" as two; nothing
// downstream could match a platform against a preference list. Every alias here was seen in the
// corpus.
std::string platformValue(std::string_view token);

// The last plausible year in the text, rejecting both halves of "1920x1080".
//
// The exclusion range is how a DATE keeps its own year. "Some.Show.2019.2023.06.01" states the
// year of the work (2019) and the day this episode went out (2023-06-01), and the last plausible
// year in that string is the 2023 inside the date — which is not the release year and must not
// be reported as one. Passing the date span in leaves the real answer as the last year outside it.
Reading yearIn(std::string_view text, std::int32_t excludeBegin = -1,
               std::int32_t excludeEnd = -1);

// A BROADCAST DATE, for the releases that number by date rather than by episode. `reading.value`
// is the ISO date, so it reads the same everywhere it is shown or logged. Only year-first forms
// and bare YYMMDD are accepted: "25-05-21" is either 2025-05-21 or 25 May 2021, and a guess
// there would be a fact invented rather than read.
struct DateReading {
    Date date;
    Reading reading;
};
DateReading dateIn(std::string_view text);

// A CRC32 AS WRITTEN: exactly eight hexadecimal characters, ignoring any brackets around them.
// Used to refuse a checksum value the characters cannot support; see the Crc32 branch in
// `mapping.cpp` for why the model needs that backstop.
bool isChecksum(std::string_view text);

} // namespace neurelease::convert
