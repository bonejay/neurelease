// From typed spans to ReleaseInfo. The model has already said WHERE each fact is and WHAT KIND it
// is; this walk says what it MEANS: each span's text goes through convert::*Value into a canonical
// value, precedence rules settle competing claims (two HDR tags, several editions, a group named
// twice), and every conversion is recorded as a FieldOrigin so the caller can trace a value back
// to its bytes. Nothing here re-scans the name -- if the model missed a span, mapping cannot
// invent it.

#include "model/mapping.hpp"

#include "convert/field_values.hpp"
#include "model/schema.hpp"
#include "text/regex.hpp"
#include "text/unicode.hpp"

#include <algorithm>
#include <charconv>
#include <string>
#include <utility>
#include <vector>

namespace neurelease::model {
namespace {

using convert::HdrFormat;

int integer(std::string_view value) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} ? result : 0;
}

int hdrPrecedence(HdrFormat format) {
    switch (format) {
    case HdrFormat::DolbyVision: return 4;
    case HdrFormat::Hdr10Plus: return 3;
    case HdrFormat::Hlg: return 2;
    case HdrFormat::Hdr10: return 1;
    case HdrFormat::Sdr: return 0;
    }
    return 0;
}

MediumKind mediumOf(ContentKind content) {
    switch (content) {
    case ContentKind::LiveActionMovie:
    case ContentKind::LiveActionSeries:
    case ContentKind::AnimatedMovie:
    case ContentKind::AnimatedSeries:
    case ContentKind::Movie:
    case ContentKind::Series: return MediumKind::Video;
    case ContentKind::Music: return MediumKind::Music;
    case ContentKind::BookDocument: return MediumKind::Book;
    case ContentKind::ComicManga: return MediumKind::Comic;
    case ContentKind::Software: return MediumKind::Software;
    case ContentKind::Game: return MediumKind::Game;
    case ContentKind::Other:
    case ContentKind::Unknown: return MediumKind::Unknown;
    }
    return MediumKind::Unknown;
}

bool containsInsensitive(std::string_view subject, std::string_view needle) {
    return text::foldKey(subject).find(text::foldKey(needle)) != std::string::npos;
}

bool appendUnique(std::vector<std::string>& values, std::string value) {
    const std::string key = text::foldKey(value);
    if (std::ranges::any_of(values, [&](const std::string& old) { return text::foldKey(old) == key; }))
        return false;
    values.push_back(std::move(value));
    return true;
}

std::string join(const std::vector<std::string>& values, std::string_view separator) {
    std::string result;
    for (const std::string& value : values) {
        if (!result.empty()) result += separator;
        result += value;
    }
    return result;
}

bool matches(std::string_view subject, const text::Regex& pattern) {
    return static_cast<bool>(pattern.match(subject));
}

bool hasMultiAudioTag(std::string_view value) {
    static const text::Regex pattern(R"(\b(?:Multi[ ._-]?Audio|MULTi(?![ ._-]?Sub))\b)", true);
    return matches(value, pattern);
}

bool isProper(std::string_view value) {
    static const text::Regex pattern(R"((?:^|[ ._\-\[(])(?:REAL[ ._-]+)?PROPER(?:FIX)?(?:$|[^A-Za-z]))", true);
    return matches(value, pattern);
}

// RERIP RIDES WITH REPACK because Sonarr's RepackRegex is `\b(repack|rerip)\d?\b` - one flag for
// both words. A rerip is the same event under a different scene name, and reading it as neither a
// repack nor a revision is what left `...RERIP.1080p...` ranked below the release it replaced.
//
// THE PAST TENSE IS THE SAME EVENT. Sonarr's regex stops at the bare word, so `REPACKED` - which
// the corpus writes 23 times - reads as nothing there. Here it raises the same flag.
bool isRepack(std::string_view value) {
    static const text::Regex pattern(
        R"((?:^|[ ._\-\[(])(?:REPACK(?:ED)?|RERIP(?:PED)?)[ ._-]?\d?(?:$|[^A-Za-z]))",
        true);
    return matches(value, pattern);
}

// THE WRITTEN REVISION NUMBER, or 0 when this token states none. Sonarr's VersionRegex
// (`\d[-._ ]?v(\d)[-._ ]|\[v(\d)\]|repack(\d)|rerip(\d)|(\d{3,4})p[._ ]v(\d)`) spelled for ONE
// token at a time: the tokenizer already breaks at every letter-digit transition, so `01v2` and
// `1080p.v2` reach this branch as a bare `v2` and the leading `\d`/`\d{3,4}p` branches collapse
// into the same one.
int statedRevisionIn(std::string_view value) {
    static const text::Regex tag(R"((?:^|[ ._\-\[(])v(\d{1,2})(?:$|[^A-Za-z0-9]))", true);
    static const text::Regex numbered(
        R"((?:^|[ ._\-\[(])(?:REPACK|RERIP|PROPER)[ ._-]?(\d)(?:$|[^A-Za-z0-9]))", true);
    int stated = 0;
    if (const text::Match match = tag.match(value)) stated = integer(match.captured(1));
    if (const text::Match match = numbered.match(value))
        stated = std::max(stated, integer(match.captured(1)));
    return stated;
}

// HOW MANY TIMES THE WORD REAL STANDS ON ITS OWN in this token. Counted rather than flagged,
// because the scene stacks it: `REAL.REAL.PROPER` is the second re-do of a botched PROPER. The
// loop restarts at the END OF THE CAPTURED WORD, not at the end of the match, so the delimiter the
// pattern consumed is still there to open the next one - `^` does not match at a start offset.
int realCountIn(std::string_view value) {
    static const text::Regex pattern(R"((?:^|[ ._\-\[(])(REAL)(?:$|[^A-Za-z]))", true);
    int count = 0;
    for (std::size_t from = 0; from <= value.size();) {
        const text::Match match = pattern.match(value, from);
        if (!match) break;
        ++count;
        from = match.capturedEnd(1);
    }
    return count;
}

bool isAiUpscale(std::string_view value) {
    // The Chinese and hyphenated spellings say the same thing: 增强 is "enhanced",
    // 生成 "generated". No word boundary before the CJK forms - PCRE2 puts no word boundary between
    // two non-ASCII characters, so requiring one would never match.
    static const text::Regex pattern(
        R"(\b(AI[ ._-]?(?:upscal\w*|enhanced?|generated?)|Topaz|upscaled?)\b|AI增强|AI生成|AI強化)", true);
    return matches(value, pattern);
}

bool isHybrid(std::string_view value) {
    static const text::Regex pattern(R"((?:^|[ ._\-/\[(,+~&;])Hybrid(?:$|[ ._\-/\]),+~&;]))", true);
    return matches(value, pattern);
}

bool isThreeD(std::string_view value) {
    static const text::Regex pattern(
        R"((?:^|[ ._\-])(3D|(?:Half|Full)[ ._-]?SBS|HSBS|SBS|(?:Half|Full)[ ._-]?OU|HOU|OU|TAB)(?:$|[ ._\-]))",
        true);
    return matches(value, pattern);
}

// Any dub tag: `Dub`, `Dubbed`, `English Dub`, `GerDub`, `Multi-Dub`.
bool isDubTag(std::string_view value) {
    static const text::Regex pattern(R"(dub(?:bed)?(?:$|[^a-z]))", true);
    return matches(value, pattern);
}

// A bare dub tag: `Dub`, `Dubbed`, `[DUB]` - nothing else in the token.
bool isBareDubTag(std::string_view value) {
    static const text::Regex pattern(R"(^[\[\(]?\s*dub(?:bed)?\s*[\]\)]?$)", true);
    return matches(value, pattern);
}

// Whether a span sits alone inside a pair of brackets or parentheses: `(01-13)`, `[ 01-12 ]`.
bool isBracketed(std::string_view name, const SegmentedSpan& span) {
    std::size_t before = static_cast<std::size_t>(span.begin);
    while (before > 0 && name[before - 1] == ' ') --before;
    std::size_t after = static_cast<std::size_t>(span.end);
    while (after < name.size() && name[after] == ' ') ++after;
    if (before == 0 || after >= name.size()) return false;
    const char open = name[before - 1];
    const char close = name[after];
    return (open == '(' && close == ')') || (open == '[' && close == ']');
}

// The text between two resolution spans that makes the second a downscale of the first:
// separators and the word "to", or an arrow.
bool isDownscaleConnective(std::string_view name, std::int32_t leftEnd, std::int32_t rightBegin) {
    if (leftEnd < 0 || rightBegin <= leftEnd || rightBegin - leftEnd > 8) return false;
    const std::string_view between = name.substr(static_cast<std::size_t>(leftEnd),
                                                 static_cast<std::size_t>(rightBegin - leftEnd));
    static const text::Regex pattern(R"(^[ ._\-]*(?:to|->|>)[ ._\-]*$)", true);
    return matches(between, pattern);
}

// `4K.to` AS ONE SPAN. The segmenter sometimes takes the connective into the resolution span
// itself, and then the text BETWEEN the two spans is a bare separator, the downscale reads as a
// mislabel, and the file is published at the resolution it was downscaled FROM. Trimming a
// trailing connective off the left span puts it back between them, where the rule looks for it.
std::int32_t resolutionValueEnd(std::string_view raw, std::int32_t end) {
    static const text::Regex trailing(R"([ ._\-]*(?:to|->|>)$)", true);
    const text::Match found = trailing.match(raw);
    if (!found) return end;
    return end - static_cast<std::int32_t>(raw.size() - found.capturedStart(0));
}

bool isRemux(std::string_view value) {
    static const text::Regex pattern(R"(\b(?:BD|UHD|Blu[ ._-]?Ray)?[ ._-]*REMUX(?=$|[ ._\-]|x26[45]))", true);
    return matches(value, pattern);
}

// ADJACENT MEANS SEPARATORS ONLY. `Blade Runner 2049` has one dot between the title and the
// number; `Blade Runner.1080p.2049` does not, and there the number is not part of the name.
bool adjacentAfter(std::string_view name, std::int32_t leftEnd, std::int32_t rightBegin) {
    if (leftEnd < 0 || rightBegin < leftEnd) return false;
    if (rightBegin - leftEnd > 3) return false;
    for (std::int32_t at = leftEnd; at < rightBegin; ++at) {
        const char character = name[static_cast<std::size_t>(at)];
        if (character != ' ' && character != '.' && character != '_' && character != '-') {
            return false;
        }
    }
    return true;
}

struct Accumulated {
    std::string audioCodec;
    std::string audioChannels;
    std::string audioProfile;
    HdrFormat hdr = HdrFormat::Sdr;
    int hdrRank = 0;
};

class Builder {
  public:
    explicit Builder(ReleaseInfo& info) : info_(info) {}

    void record(Field field, std::string value, const SegmentedSpan& span, std::string_view raw) {
        info_.origins.push_back({field, std::move(value), std::string(raw), span.begin, span.end,
                                 span.confidence, false});
        info_.origins.back().unconverted = info_.origins.back().value.empty();
    }

    void verdict(Field field, const SegmentedField& answer) {
        if (answer.value.empty()) return;
        info_.origins.push_back({field, answer.value, {}, -1, -1, answer.confidence, false});
    }

  private:
    ReleaseInfo& info_;
};

} // namespace

ReleaseInfo releaseInfoFromAnalysis(std::string_view name, const Analysis& analysis) {
    ReleaseInfo info;
    info.rawName = std::string(text::trimmed(name));
    if (!analysis.valid) return info;

    const ContentKind content = analysis.content == ContentKind::Unknown
                                    ? schema::contentKind(analysis.contentKind.value) : analysis.content;
    NumberingKind numbering = analysis.numbering == NumberingKind::Unknown
                                  ? schema::numberingKind(analysis.numberingKind.value) : analysis.numbering;
    const PackScope packScope = analysis.pack == PackScope::Unknown
                                    ? schema::packScope(analysis.packScope.value) : analysis.pack;
    const SpecialKind special = analysis.special == SpecialKind::Unknown
                                    ? schema::specialKind(analysis.specialKind.value) : analysis.special;
    const AdultKind adult = analysis.adult == AdultKind::Unknown
                                ? schema::adultKind(analysis.adultKind.value) : analysis.adult;

    info.content = content;
    info.numbering = numbering;
    info.packScope = packScope;
    info.special = special;
    info.adult = adult;
    info.contentConfidence = analysis.contentKind.confidence;
    info.numberingConfidence = analysis.numberingKind.confidence;
    info.packScopeConfidence = analysis.packScope.confidence;
    info.specialConfidence = analysis.specialKind.confidence;
    info.adultConfidence = analysis.adultKind.confidence;
    // The one distinction a release name makes about animation. See ReleaseInfo::anime.
    info.anime = analysis.anime;
    info.animeConfidence = analysis.animeKind.confidence;

    Builder builder(info);
    Accumulated accumulated;
    bool sourceWasBareUhd = false;
    bool seasonSeen = false;
    bool episodeSeen = false;
    // Where the last resolution span ended, for the downscale reading above.
    std::int32_t resolutionEnd = -1;
    // WHERE THE TITLE ENDED, so a following number can be tested for adjacency to it rather than
    // merely for coming after it somewhere in the name.
    std::int32_t titleEnd = -1;
    std::string pendingTitleNumber;
    SegmentedSpan pendingTitleSpan{};
    // ONE SPAN STATED BOTH. Only that shape can contradict an absolute verdict, and the flag is
    // what keeps the correction below from touching the many names where a season marker sits
    // beside genuinely absolute numbering.
    bool combinedMarkerSeen = false;
    std::int32_t episodeTitleEnd = -1;
    // THE REVISION IS SPELLED ACROSS SEVERAL SPANS, so it cannot be decided inside one of them.
    // `REAL.REAL.PROPER` and `PROPER` beside `v2` arrive here as separate edition tokens, and the
    // version is the highest number ANY of them wrote plus one if ANY of them was a proper or a
    // repack - so the two halves of that sum are carried across the walk and added once at the end.
    int revisionStated = 0;
    bool revisionBumped = false;

    for (const SegmentedSpan& span : analysis.spans) {
        if (span.begin < 0 || span.end <= span.begin || static_cast<std::size_t>(span.end) > name.size())
            continue;
        const std::string raw = std::string(text::trimmed(name.substr(
            static_cast<std::size_t>(span.begin), static_cast<std::size_t>(span.end - span.begin))));
        if (raw.empty()) continue;
        const SpanType type = span.type;

        if (type == SpanType::Main) {
            std::string title = convert::titleText(raw);
            // A number held from a rejected year, sitting immediately in front of this title.
            if (!pendingTitleNumber.empty() && adjacentAfter(name, pendingTitleSpan.end, span.begin)) {
                title = pendingTitleNumber + " " + title;
            }
            pendingTitleNumber.clear();
            if (info.title.empty()) {
                info.title = title;
                titleEnd = span.end;
                builder.record(Field::Title, title, span, raw);
            } else {
                info.alternativeTitles.push_back(title);
                builder.record(Field::AlternateTitle, title, span, raw);
            }
        } else if (type == SpanType::Alternate) {
            const std::string title = convert::titleText(raw);
            info.alternativeTitles.push_back(title);
            builder.record(Field::AlternateTitle, title, span, raw);
        } else if (type == SpanType::EpisodeTitle) {
            // Read like every other title field: separators become spaces, the raw spelling stays
            // beside it as the origin text. The first one is the field; a second is evidence only.
            const std::string title = convert::titleText(raw);
            if (info.episodeTitle.empty()) {
                info.episodeTitle = title;
                episodeTitleEnd = span.end;
            }
            builder.record(Field::EpisodeTitle, title, span, raw);
        } else if (type == SpanType::Year) {
            const convert::DateReading date = convert::dateIn(raw);
            if (date.date.valid()) {
                info.date = date.date;
                builder.record(Field::AirDate, date.reading.value, span, raw);
            } else {
                const std::string year = convert::yearIn(raw).value;
                const int value = integer(year);
                // A NUMBER TOO LARGE TO BE A YEAR IS TITLE TEXT. `Blade Runner 2049` has no local
                // signal that 2049 belongs to the title - it looks exactly like a year, and the
                // model reads it as one at p 0.97 - except that nothing is released in 2049. So
                // the span is given back to the text beside it: the title before it when the two
                // are adjacent, otherwise the title that follows.
                //
                // BEFORE WINS when both are adjacent, because a trailing number is part of the
                // name in `Blade Runner 2049` and `Cyberpunk 2077` while a leading one is an
                // episode number far more often than it is part of a title.
                //
                // The upper bound is deliberately generous and fixed rather than clock-based: a
                // runtime whose answers change with the date is not reproducible, and a name
                // stating a year a decade out is stating something else.
                // NO YEAR PARSED IS NOT AN IMPLAUSIBLE YEAR. `odc.2770-2774` is a span the model
                // called a year and `yearIn` reads nothing from, because 2770 is not year-shaped
                // at all; treating that as out of range appended the episode numbers to the title
                // and made it worse. Only a number that IS year-shaped and outside the range is
                // title text.
                if (year.empty()) {
                    builder.record(Field::Year, "", span, raw);
                } else if (value >= convert::PlausibleYearFirst
                           && value <= convert::PlausibleYearLast) {
                    info.year = value;
                    builder.record(Field::Year, year, span, raw);
                } else if (!info.title.empty() && adjacentAfter(name, titleEnd, span.begin)) {
                    info.title += ' ';
                    info.title += convert::titleText(raw);
                    titleEnd = span.end;
                    builder.record(Field::Title, info.title, span, raw);
                } else {
                    // Nothing to attach it to yet. Held for the next title span, which is the
                    // `2009.shoot.fruit.chan` shape - a number in front of the name.
                    pendingTitleNumber = convert::titleText(raw);
                    pendingTitleSpan = span;
                }
            }
        } else if (type == SpanType::Resolution || type == SpanType::FrameSize) {
            const ResolutionTier tier = convert::resolutionValue(raw);
            // `4K.to.1080p`: two resolutions with "to" between them state a downscale, and the
            // file is the SECOND one. Without the connective the higher tier wins as before -
            // `2160p.1080p` in one name is a mislabel, not a statement.
            const bool downscale = resolutionEnd >= 0 && tier != ResolutionTier::Unknown
                                   && static_cast<std::uint8_t>(tier) < static_cast<std::uint8_t>(info.screenSize)
                                   && isDownscaleConnective(name, resolutionEnd, span.begin);
            if (downscale) {
                info.screenSize = tier;
                info.downscaled = true;
                builder.record(Field::Downscaled, "yes", span, raw);
            } else if (static_cast<std::uint8_t>(tier) > static_cast<std::uint8_t>(info.screenSize)) {
                info.screenSize = tier;
            }
            resolutionEnd = resolutionValueEnd(raw, span.end);
            builder.record(Field::Quality, std::string(label(tier)), span, raw);
        } else if (type == SpanType::SourceType) {
            const SourceKind source = convert::sourceValue(raw);
            const bool bareUhd = convert::sourceTokenIsBareUhd(raw);
            if (source != SourceKind::Unknown &&
                (info.source == SourceKind::Unknown || (sourceWasBareUhd && !bareUhd))) {
                info.source = source;
                sourceWasBareUhd = bareUhd;
            }
            const bool remux = convert::sourceTokenIsRemux(raw);
            const bool light = convert::sourceTokenIsLightEncode(raw);
            info.remux = info.remux || remux;
            info.lightEncode = info.lightEncode || light;
            // A SOURCE TOKEN THAT STATES OTHER FIELDS TOO - `UHDRDV` is 2160p with HDR10 and
            // Dolby Vision in one word. Applied here rather than by widening sourceValue, because
            // these are facts about OTHER fields and folding them into a source value would lose
            // them. A stated span always wins: the resolution is only filled when nothing else
            // gave one, while the HDR flags are additive exactly as a stated `DV.HDR10` is.
            const convert::SourceTokenExtras extras = convert::sourceTokenExtras(raw);
            if (extras.any()) {
                if (extras.screenSize != ResolutionTier::Unknown &&
                    info.screenSize == ResolutionTier::Unknown)
                    info.screenSize = extras.screenSize;
                if (extras.hdr10) info.hdr10 = true;
                if (extras.dolbyVision) info.dolbyVision = true;
                if (extras.hdr10 || extras.dolbyVision) {
                    const HdrFormat format = extras.dolbyVision ? HdrFormat::DolbyVision
                                                                : HdrFormat::Hdr10;
                    const int rank = hdrPrecedence(format);
                    if (rank > accumulated.hdrRank) {
                        accumulated.hdrRank = rank;
                        accumulated.hdr = format;
                    }
                }
            }
            builder.record(Field::ReleaseSource,
                           source != SourceKind::Unknown ? std::string(label(source))
                           : remux ? "remux" : light ? "light encode" : "",
                           span, raw);
        } else if (type == SpanType::Platform) {
            const std::string platform = convert::platformValue(raw);
            if (info.streamingService.empty()) info.streamingService = platform;
            builder.record(Field::Platform, platform, span, raw);
        } else if (type == SpanType::Codec) {
            const VideoCodec codec = convert::codecValue(raw);
            if (codec != VideoCodec::Unknown && info.videoCodec == VideoCodec::Unknown)
                info.videoCodec = codec;
            // A CODEC TOKEN CAN STATE TWO FACTS. `Hevc10`, `x265-10bit` and `H264.Hi10P` name the
            // codec and the depth in one run, and the model gets one label per segment: on
            // `Hevc10` it ranked codec 0.79 against bit depth 0.18, so the depth was seen and
            // then lost. Read both rather than making the segmenter choose.
            if (const std::string depth = convert::bitDepthIn(raw).value; depth == "10bit")
                info.tenBit = true;
            builder.record(Field::Codec, std::string(label(codec)), span, raw);
        } else if (type == SpanType::BitDepth) {
            const std::string depth = convert::bitDepthIn(raw).value;
            if (depth == "10bit") info.tenBit = true;
            // AND THE MIRROR OF THE CODEC BRANCH ABOVE. `Hi10P` is the H.264 High 10 profile and
            // `Main10` an HEVC profile: whichever label the model puts on the span, the token
            // states a codec as well, and reading it only in one branch loses it in the other.
            if (const VideoCodec named = convert::codecValue(raw);
                named != VideoCodec::Unknown && info.videoCodec == VideoCodec::Unknown)
                info.videoCodec = named;
            std::string value = depth;
            if (value.empty() && convert::statesAudioSampleDepth(raw)) {
                value = "audio sample depth";
                if (info.medium == MediumKind::Unknown) info.medium = MediumKind::Music;
            } else if (value.empty() && convert::statesSoftwareBitness(raw)) {
                value = "software bitness";
                if (info.medium == MediumKind::Unknown) info.medium = MediumKind::Software;
            }
            builder.record(Field::BitDepth, value, span, raw);
        } else if (type == SpanType::Hdr) {
            const convert::HdrReading hdr = convert::hdrValue(raw);
            const int rank = hdrPrecedence(hdr.format);
            if (rank > accumulated.hdrRank) {
                accumulated.hdrRank = rank;
                accumulated.hdr = hdr.format;
            }
            // EVERY stated format sets its flag. The precedence above chooses the one-string
            // summary; the booleans are documented as the full set, because a DV release
            // routinely states an HDR10 base layer beside it and "DV.HDR10" means both.
            switch (hdr.format) {
            case HdrFormat::DolbyVision: info.dolbyVision = true; break;
            case HdrFormat::Hdr10Plus: info.hdr10Plus = true; break;
            case HdrFormat::Hlg: info.hlg = true; break;
            case HdrFormat::Hdr10: info.hdr10 = true; break;
            default: break;
            }
            builder.record(Field::Hdr, hdr.reading.value, span, raw);
        } else if (type == SpanType::AudioCodec) {
            // `DD+` IS A DIFFERENT CODEC FROM `DD`, AND THE PLUS IS OUTSIDE THE SPAN. The
            // tokeniser breaks on `+`, so `DD+5.1` is read as an audio codec `DD` beside a
            // channel layout `5.1` - correctly - and the one character that distinguishes Dolby
            // Digital Plus from Dolby Digital never reaches the converter. `audioCodecValue`
            // already tests for `DD+`; that branch was simply unreachable. 18,085 names in the
            // 23.7M-name dump write it this way, against `DDP` which already works.
            //
            // Reading ONE character past the span, and only a literal `+`, is decidable from the
            // characters alone and cannot change any other reading.
            std::string spelling = raw;
            if (static_cast<std::size_t>(span.end) < name.size() && name[span.end] == '+')
                spelling.push_back('+');
            const std::string codec = convert::audioCodecValue(spelling);
            if (accumulated.audioCodec.empty()) accumulated.audioCodec = codec;
            builder.record(Field::Audio, codec, span, raw);
        } else if (type == SpanType::AudioChannels) {
            const std::string channels = convert::audioChannelsValue(raw);
            if (accumulated.audioChannels.empty()) accumulated.audioChannels = channels;
            builder.record(Field::Audio, channels, span, raw);
        } else if (type == SpanType::AudioFeature) {
            // THE FEATURE IS A VALUE, NOT A TEST FOR ONE WORD. This branch used to ask only
            // whether the span said `ATMOS` and drop it otherwise, so `DTS-X` was located,
            // labelled at 0.91 by the model, and then thrown away - a fact the documentation
            // already promised. The first feature wins, as elsewhere.
            const std::string profile = convert::audioProfileValue(raw);
            if (accumulated.audioProfile.empty()) accumulated.audioProfile = profile;
            builder.record(Field::Audio, profile, span, raw);
        } else if (type == SpanType::AudioLanguage) {
            const std::vector<std::string> languages = convert::languageCodesOfToken(raw);
            for (const std::string& language : languages) appendUnique(info.languages, language);
            if (languages.empty() && hasMultiAudioTag(raw)) info.multiAudio = true;
            // `[Dub]`, `Dubbed`, `English Dub`, `GerDub`: a dub tag says a dubbed track is present.
            // A bare one, in the anime naming it comes from, is the English dub; one that names
            // its language is that language's dub, and is the English dub only when it says so.
            const bool bareDub = languages.empty() && isBareDubTag(raw);
            const bool namedEnglishDub = isDubTag(raw)
                && std::ranges::find(languages, "eng") != languages.end();
            if (bareDub || namedEnglishDub) info.englishDub = true;
            builder.record(Field::AudioLanguage,
                           !languages.empty() ? join(languages, "/")
                           : info.multiAudio ? "multi audio"
                           : bareDub ? "english dub" : "",
                           span, raw);
        } else if (type == SpanType::SubtitleLanguage) {
            std::vector<std::string> languages = convert::languageCodesOfToken(raw);
            static const text::Regex englishSubs(R"(\bE[ ._-]?Subs?\b)", true);
            if (languages.empty() && englishSubs.match(raw)) languages.push_back("eng");
            for (const std::string& language : languages) appendUnique(info.subtitleLanguages, language);
            const bool several = convert::statesSeveralSubtitles(raw);
            const bool present = several || convert::statesSubtitlesOnly(raw);
            if (several) info.multiSubs = true;
            const SubtitleFormat format = convert::subtitleFormatValue(raw);
            if (format != SubtitleFormat::Unknown && info.subtitleFormat == SubtitleFormat::Unknown)
                info.subtitleFormat = format;
            const std::string value = !languages.empty() ? join(languages, "/")
                                      : format != SubtitleFormat::Unknown ? std::string(label(format))
                                      : several ? "several" : present ? "subtitles" : "";
            builder.record(Field::SubtitleLanguage, value, span, raw);
            if (format != SubtitleFormat::Unknown)
                builder.record(Field::SubtitleFormat, std::string(label(format)), span, raw);
        } else if (type == SpanType::DualAudioTag) {
            info.dualAudio = true;
            if (containsInsensitive(raw, "dub") && containsInsensitive(raw, "eng")) info.englishDub = true;
            builder.record(Field::DualAudio, "yes", span, raw);
        } else if (type == SpanType::HardsubTag) {
            static const text::Regex negated(R"(\bwithout\b|\bno\b)", true);
            static const text::Regex softOnly(R"(\bsoft[ ._-]?subs?\b)", true);
            const bool burned = !negated.match(raw) && !softOnly.match(raw);
            info.hardSubs = info.hardSubs || burned;
            const std::vector<std::string> languages = convert::languageCodesOfToken(raw);
            for (const std::string& language : languages) appendUnique(info.subtitleLanguages, language);
            builder.record(Field::HardSubs, burned ? (languages.empty() ? "yes" : join(languages, "/")) : "no",
                           span, raw);
        } else if (type == SpanType::FranchisePrefix) {
            // RECORDED, NOT DROPPED. This branch did not exist: the segmenter marked the span and
            // the mapping ignored it, so `007.James.Bond.1963.From.Russia.With.Love` reached a
            // caller with the prefix nowhere — neither in the title nor as an origin it could
            // trace. Recorded as its own field so a caller can strip it, keep it, or ignore it,
            // and so the benchmark can score the shape at all.
            const std::string prefix = convert::titleText(raw);
            if (info.franchisePrefix.empty()) info.franchisePrefix = prefix;
            builder.record(Field::FranchisePrefix, prefix, span, raw);
        } else if (type == SpanType::SubtitlePart) {
            // RETIRED IN THE LABELLING, STILL SPOKEN BY OLDER WEIGHTS. No model trained after
            // 2026-09-15 emits this type: a title's subtitle is title text, and the trainer reads
            // every older label that way. A checkpoint from before still says it, and it is read
            // here exactly as the trainer reads the gold - as the title's own text, in place, and
            // never as a field of its own. Appending every part to the END of the title, as this
            // did, put `Sword Art Online Alicization` together correctly and `Alicization Sword
            // Art Online` in the wrong order when the subtitle stood first.
            const std::string part = convert::titleText(raw);
            if (!info.title.empty() && adjacentAfter(name, titleEnd, span.begin)) {
                // Joined to the title before it: the title grows.
                info.title += ' ';
                info.title += part;
                titleEnd = span.end;
                builder.record(Field::Title, info.title, span, raw);
            } else if (info.title.empty()) {
                // Written before its title: held for the next title span, the way a number the
                // year reader gave back is held.
                pendingTitleNumber = part;
                pendingTitleSpan = span;
            } else if (!info.episodeTitle.empty() && adjacentAfter(name, episodeTitleEnd, span.begin)) {
                // Beside an episode title and not glued to the main title: the episode title's
                // own text. (A subtitle written BEFORE its episode title is not caught here -
                // this reader is one pass, and only older weights say the type at all.)
                info.episodeTitle += ' ';
                info.episodeTitle += part;
                episodeTitleEnd = span.end;
                builder.record(Field::EpisodeTitle, info.episodeTitle, span, raw);
            } else {
                // Something stands between it and the title. The trainer calls that a title-like
                // text outside the main span - an alternate title - and so does this.
                info.alternativeTitles.push_back(part);
                builder.record(Field::AlternateTitle, part, span, raw);
            }
        } else if (type == SpanType::ReleaseGroup) {
            const std::string group = convert::groupText(raw);
            const bool usable = !group.empty() && !convert::isNeverAGroup(group);
            if (usable) appendUnique(info.releaseGroups, group);
            if (usable && info.releaseGroup.empty()) info.releaseGroup = group;
            builder.record(Field::Group, usable ? group : std::string{}, span, raw);
        } else if (type == SpanType::SeasonMarker) {
            const convert::MarkerNumbers numbers = convert::markerNumbersIn(raw);
            if (numbers.stated && !seasonSeen) {
                seasonSeen = true;
                info.season = numbers.first;
                if (numbers.last > numbers.first) {
                    info.seasonEnd = numbers.last;
                    info.pack = true;
                }
            }
            builder.record(Field::Season, numbers.stated ? std::to_string(numbers.first) : "", span, raw);
        } else if (type == SpanType::EpisodeMarker) {
            const convert::EpisodeMarkerReading numbers = convert::episodeMarkerIn(raw);
            const int first = numbers.first;
            const int last = numbers.last;
            const int count = numbers.count;
            const int crossSeason = numbers.season;
            if (numbers.stated && !episodeSeen) {
                episodeSeen = true;
                info.episode = first;
                if (last > first) info.episodeEnd = last;
                if (crossSeason > 0 && !seasonSeen) {
                    seasonSeen = true;
                    info.season = crossSeason;
                }
                if (last > first) {
                    info.pack = true;
                    // THE RANGE STATES ITS COUNT. `E01-E13` is thirteen episodes, not the two
                    // numbers it took to write them; an explicit count in the marker still wins
                    // when it is larger.
                    info.episodeCount = std::max(count, last - first + 1);
                    // A BRACKETED RANGE IS THE FANSUB BATCH: `(01-13)`, `[01-12]`. It states the
                    // whole of what the uploader had - a cour - and the aggregate totals metadata
                    // holds for a split-cour series are routinely wrong beside it, so a consumer
                    // may treat it as complete. Ten or more, as the convention has always been
                    // read: `(01-03)` is three episodes of something, not a season of them.
                    if (last >= 10 && isBracketed(name, span)) info.explicitCompleteRange = true;
                }
            } else if (numbers.stated && episodeSeen && count == 1 && last == 0) {
                // A SECOND MARKER THAT CONTINUES THE FIRST. `S01E02.S01E03.S01E04` and
                // `1x02 - 1x03 - 1x04` state a range across several markers, and only the range
                // written inside ONE marker was being read - so the name said three episodes and
                // the result said one. The model marks each of them correctly; this is the
                // conversion layer joining what it marked, not a second opinion about the spans.
                //
                // Only the next number extends it. `Episode 38 (152)` is a relative number beside
                // an absolute one, not a range, and every enumeration in the corpora is
                // consecutive - so requiring exactly `+1` takes the real ones and leaves that
                // alone. A stated season must agree too, or `S01E02.S02E01` would fuse.
                const int currentEnd = info.episodeEnd ? *info.episodeEnd : *info.episode;
                const bool sameSeason = crossSeason == 0 || !info.season
                                        || crossSeason == *info.season;
                if (sameSeason && first == currentEnd + 1) {
                    info.episodeEnd = first;
                    info.pack = true;
                    info.episodeCount = *info.episodeEnd - *info.episode + 1;
                }
            }
            builder.record(Field::Episode, numbers.stated ? std::to_string(first) : "", span, raw);
        } else if (type == SpanType::SeasonEpisodeMarker) {
            // ONE SPAN, BOTH FIELDS, and the only span type that fills two. The model decided this
            // run of digits states a season and an episode; the split is then arithmetic. Recorded
            // as two origins so `stated` and the C ABI see a season and an episode exactly as they
            // would from two separate markers - nothing downstream needs to know it came from one.
            const convert::EpisodeMarkerReading numbers = convert::seasonEpisodeMarkerIn(raw);
            if (numbers.stated) {
                if (!seasonSeen && numbers.season > 0) {
                    seasonSeen = true;
                    combinedMarkerSeen = true;
                    info.season = numbers.season;
                    builder.record(Field::Season, std::to_string(numbers.season), span, raw);
                }
                if (!episodeSeen) {
                    episodeSeen = true;
                    info.episode = numbers.first;
                    if (numbers.last > numbers.first) {
                        info.episodeEnd = numbers.last;
                        info.pack = true;
                        info.episodeCount = std::max(numbers.count, numbers.last - numbers.first + 1);
                    }
                    builder.record(Field::Episode, std::to_string(numbers.first), span, raw);
                }
            }
        } else if (type == SpanType::VolumeMarker) {
            info.pack = true;
            builder.record(Field::PackMarker, raw, span, raw);
        } else if (type == SpanType::PackMarker) {
            info.pack = true;
            if (containsInsensitive(raw, "complete")) info.explicitCompleteRange = true;
            builder.record(Field::PackMarker, raw, span, raw);
        } else if (type == SpanType::SpecialMarker) {
            info.specials = true;
            builder.record(Field::SpecialMarker, raw, span, raw);
        } else if (type == SpanType::Edition) {
            for (EditionKind edition : convert::editionsIn(raw))
                if (std::ranges::find(info.editions, edition) == info.editions.end())
                    info.editions.push_back(edition);
            const EditionKind edition = convert::editionIn(raw);
            if (edition != EditionKind::Unknown && info.edition == EditionKind::Unknown)
                info.edition = edition;
            std::vector<std::string> routed;
            if (isProper(raw)) { info.proper = true; routed.push_back("proper"); }
            if (isRepack(raw)) { info.repack = true; routed.push_back("repack"); }
            // THE REVISION, GATHERED HERE AND SUMMED AFTER THE WALK. A proper or a repack raises
            // the version whether or not it wrote a number; REAL never does, so it is only counted.
            revisionStated = std::max(revisionStated, statedRevisionIn(raw));
            if (isProper(raw) || isRepack(raw)) revisionBumped = true;
            info.revisionReal += realCountIn(raw);
            if (isAiUpscale(raw)) { info.aiUpscale = true; routed.push_back("ai upscale"); }
            if (isHybrid(raw)) { info.hybrid = true; routed.push_back("hybrid"); }
            if (isThreeD(raw)) { info.threeD = true; routed.push_back("3D"); }
            if (isRemux(raw)) { info.remux = true; routed.push_back("remux"); }
            builder.record(Field::Edition, edition != EditionKind::Unknown
                                               ? std::string(label(edition)) : join(routed, "+"),
                           span, raw);
        } else if (type == SpanType::SiteBanner) {
            builder.record(Field::SiteBanner, raw, span, raw);
        } else if (type == SpanType::TrackerTag) {
            builder.record(Field::TrackerTag, raw, span, raw);
        } else if (type == SpanType::Container || type == SpanType::FileExtension) {
            const std::string container = convert::containerValue(raw);
            if (!container.empty()) {
                info.container = container;
                info.medium = convert::mediumOfContainer(container);
                if (info.medium == MediumKind::Subtitle &&
                    info.subtitleFormat == SubtitleFormat::Unknown)
                    info.subtitleFormat = convert::subtitleFormatValue(container);
            }
            builder.record(Field::Container, container, span, raw);
        } else if (type == SpanType::Crc32) {
            // A CRC32 IS EIGHT HEXADECIMAL CHARACTERS. Nothing else is one, so a span the model
            // calls a checksum but that cannot be one is recorded WITHOUT a value rather than
            // published as a checksum. The model proposes, the value converter refuses - the same
            // arrangement `isBareResolution` already uses to keep `720` out of the episode field.
            //
            // WHY THIS IS WORTH A GATE. `crc32` is where the type head puts short strings it does
            // not recognise: measured over the corpus discrepancy set it is ranked first on
            // `READNFO`, `QC`, `NTSC`, `LD`, `WS`, `US`, `pal`, `XXX`, `Chaps`, `LAME3*92` and -
            // at p 0.72 - the actor name `Steven Seagal`. In 3,368 gold checksum spans, 3,355 are
            // exactly eight hex characters; every one of the other thirteen is a truncated label
            // (`2575F9D`, `811C87827`, `981`), so nothing correct is refused by this test.
            //
            // MD5 (32) and SHA-1 (40) runs do occur in the wild - about 1,400 and 50 per two
            // million names - but they are not CRC32s and there is no field for them, so they are
            // dropped here rather than published under a name that would be wrong.
            if (!convert::isChecksum(raw) && isThreeD(raw)) {
                // `3D`, `Half-SBS`, `HSBS`: short tags the type head files here as the class it
                // uses for text it does not recognise. They are one thing the app must not miss -
                // a 3D rip is never the release anyone wants - so a refused checksum that reads as
                // a 3D layout is one, and is recorded as one.
                info.threeD = true;
                builder.record(Field::ThreeD, "yes", span, raw);
            } else {
                builder.record(Field::Crc32, convert::isChecksum(raw) ? raw : std::string{}, span, raw);
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // RULES THAT DO NOT COME FROM THE MODEL. Everything above this line reads spans the segmenter
    // produced. What follows re-reads the NAME, and that is a different and worse kind of code, so
    // it lives here in one block rather than scattered through the walk above.
    //
    // WE DO NOT LIKE THIS AND IT SHOULD NOT GROW. The whole point of a learned segmenter is that
    // notation is its problem: the model is asked which characters are a season and which are an
    // episode, and a hand-written pattern that answers the same question is the thing this parser
    // exists to avoid. Every rule below is an admission that the model got a decidable case wrong.
    // Three constraints keep that admission honest:
    //
    //   1. A STATED SPAN ALWAYS WINS. Nothing here overrides the model; each rule fills a field
    //      the model left empty, so a better segmenter silently retires it.
    //   2. IT MUST BE DECIDABLE FROM THE CHARACTERS ALONE. `S00E01` is season 0 episode 1 in every
    //      name that has ever been written. A rule that needs to know what the work is, or that
    //      guesses at a boundary, does not belong here - it belongs in the labels.
    //   3. IT IS MEASURED, and the count is written beside it.
    //
    // A rule that cannot meet all three is a training problem wearing a regex. The remaining
    // corpus failures that look like candidates - a title with a number glued to it, an episode
    // title split at a marker word, `Cap.102` on a stated season - are all in that category and
    // are deliberately NOT here.
    //
    // TURNED DOWN ON PURPOSE, so the next reader does not re-propose them: stripping an upload
    // marker off a group name (`-PLUTONiUM-xpost` -> PLUTONiUM, 5 corpus names) and inventing a
    // title when the model returns none. The first is a word list dressed up as parsing and the
    // segmenter already splits the same thing when it is written with dots; the second had no
    // evidence behind it at all - of the names in the validation set that come back with no title,
    // not one carries another span a title could honestly be promoted from.
    //
    // A LITERAL SxxExx IS NOT THE MODEL'S TO MISS. Season and episode came only from spans the
    // segmenter marked, so a name whose `S00E01` it failed to mark lost BOTH fields — measured on
    // 26 real `S00` names from the corpus, the shipped segmenter dropped 8 of them and a candidate
    // dropped 12, in each case names whose numbering is written out unambiguously. The pattern is
    // decidable from the characters, so it is read here rather than left to a model to relearn.
    //
    // Recorded as ORIGINS, not just as values: `stated` is derived from the origins, which is what
    // lets a written zero (`S00`, `E00`) be told apart from an absent field, and a value without
    // an origin would read as absent through the C ABI. A stated span always wins — this fills
    // only what the model left empty.
    // A NAME WITH TITLES BUT NO MAIN ONE. `title` comes only from a span the segmenter typed as
    // the main reading, so a name whose title spans it typed ALL as alternates published no title
    // at all: `[Ironclad] Kowloon Generic Romance - S01 [BD.1080p.AV1] | Kowloon Generic Romance
    // (2025) ...` came back with two identical alternates and an empty title, and an application
    // that matches a request against the title dropped the release entirely.
    //
    // Promoting the first alternate is not a guess about which reading is best - it is the same
    // order the segmenter emitted, and `alternative_title` already means "another complete title
    // of this work". A stated main span always wins; this only fills an empty field.
    //
    // Measured: 77 of 59,999 labelled names gain a title this way (3,553 without the rule, 3,476
    // with it), and 5 of the 4,182 consensus validation names.
    if (info.title.empty() && !info.alternativeTitles.empty()) {
        info.title = info.alternativeTitles.front();
        info.alternativeTitles.erase(info.alternativeTitles.begin());
        for (FieldOrigin& origin : info.origins) {
            if (origin.field != Field::AlternateTitle) continue;
            origin.field = Field::Title;
            titleEnd = origin.end;
            break;
        }
    }

    // A 3D LAYOUT TAG THE SEGMENTER STOPPED MARKING. `3D`, `Half-SBS`, `HSBS`, `Half-OU`, `HOU`:
    // model 2 filed these as an edition or as a refused checksum, and both routes set the flag;
    // model 3 returns no span for them at all, so `Example.Movie.2024.3D.Half-SBS.1080p.BluRay`
    // came back with nothing to reject. A 3D rip is never the release anyone wants, and an
    // application that silently accepts one has a worse failure than a missed field.
    //
    // Decidable from the characters, and searched only AFTER the title, because `3D Kanojo` and
    // `Tokyo 3D` are titles and not layouts. The bare `SBS`, `OU` and `TAB` spellings that the
    // edition route accepts are deliberately NOT here: `SBS` is a Korean broadcaster that appears
    // as a platform in real names, and two letters outside a span the model vouched for is a
    // coin toss. Measured on 59,999 labelled names: 32 carry the flag and all 32 come from this
    // rule, because model 3 marks none of them; 2 of the 4,182 consensus validation names.
    if (!info.threeD && titleEnd >= 0 && static_cast<std::size_t>(titleEnd) < name.size()) {
        static const text::Regex layout(
            R"((?:^|[ ._\-])(3D|(?:Half|Full)[ ._-]?SBS|HSBS|(?:Half|Full)[ ._-]?OU|HOU)(?=$|[ ._\-]))",
            true);
        const std::string_view tail = name.substr(static_cast<std::size_t>(titleEnd));
        if (const text::Match found = layout.match(tail)) {
            SegmentedSpan layoutSpan;
            layoutSpan.type = SpanType::Edition;
            layoutSpan.begin = titleEnd + static_cast<std::int32_t>(found.capturedStart(1));
            layoutSpan.end = titleEnd + static_cast<std::int32_t>(found.capturedEnd(1));
            layoutSpan.confidence = 1.0F;
            info.threeD = true;
            builder.record(Field::ThreeD, "yes", layoutSpan, found.captured(1));
        }
    }

    if (!seasonSeen || !episodeSeen) {
        static const text::Regex writtenOut(
            R"((?:^|[^A-Za-z0-9])(s(\d{1,2}))[ ._-]?(e(\d{1,4})))", true);
        if (const text::Match found = writtenOut.match(name)) {
            SegmentedSpan seasonSpan;
            seasonSpan.type = SpanType::SeasonMarker;
            seasonSpan.begin = static_cast<std::int32_t>(found.capturedStart(1));
            seasonSpan.end = static_cast<std::int32_t>(found.capturedEnd(1));
            seasonSpan.confidence = 1.0F;
            SegmentedSpan episodeSpan;
            episodeSpan.type = SpanType::EpisodeMarker;
            episodeSpan.begin = static_cast<std::int32_t>(found.capturedStart(3));
            episodeSpan.end = static_cast<std::int32_t>(found.capturedEnd(3));
            episodeSpan.confidence = 1.0F;
            if (!seasonSeen) {
                seasonSeen = true;
                info.season = integer(found.captured(2));
                builder.record(Field::Season, std::to_string(*info.season), seasonSpan,
                               found.captured(1));
            }
            if (!episodeSeen) {
                episodeSeen = true;
                info.episode = integer(found.captured(4));
                builder.record(Field::Episode, std::to_string(*info.episode), episodeSpan,
                               found.captured(3));
            }
        }
    }

    if (!info.year && info.date.valid()) info.year = info.date.year;
    // THE REVISION, SUMMED ONCE THE WHOLE NAME HAS BEEN SEEN. A stated number alone IS the
    // version; a proper or a repack makes it that number plus one, or 2 when nothing was written,
    // because the PROPER is itself the second issue of the release. Nothing stated leaves the
    // default 1: an unstated revision is the first one, not a missing one.
    info.revisionVersion = revisionStated > 0 ? revisionStated : 1;
    if (revisionBumped) info.revisionVersion = revisionStated > 0 ? revisionStated + 1 : 2;
    // The flags were set per stated span above; the summary alone still implies its own flag,
    // so a name that stated exactly one format keeps behaving as before.
    info.dolbyVision |= accumulated.hdr == HdrFormat::DolbyVision;
    info.hdr10Plus |= accumulated.hdr == HdrFormat::Hdr10Plus;
    info.hlg |= accumulated.hdr == HdrFormat::Hlg;
    info.hdr10 |= accumulated.hdr == HdrFormat::Hdr10;
    info.hdr = std::string(convert::hdrFormatLabel(accumulated.hdr));
    info.audioCodec = convert::audioCodecValue(accumulated.audioCodec);
    info.audioChannels = convert::audioChannelsValue(accumulated.audioChannels);
    info.audioProfile = accumulated.audioProfile;
    if (info.languages.size() > 2) info.multiAudio = true;

    if (info.medium == MediumKind::Unknown) info.medium = mediumOf(content);
    // A COMBINED MARKER OVERRULES AN ABSOLUTE VERDICT, and nothing else does. `Show Name 804`
    // came out as season 8 with absolute episode 4 and no episode at all - a season holding
    // nothing, which is nobody's reading. One span asserting both numbers contradicts a
    // series-wide reading of the same digits outright, so it wins.
    //
    // A SEASON MARKER ALONE DOES NOT, which the labels are clear about and a first attempt at
    // this got wrong. `[Erai-raws] Tondemo Skill ... 2 - 01`, `Ace of the Diamond ~Second
    // Season~ - 32` and `べっぴんさん 第23週「...」#128-133` all pair a season with genuinely
    // absolute numbering, and the gold reads every one of them as absolute. Anime and Japanese
    // broadcast TV do this routinely; overruling there replaced one wrong answer with another.
    //
    // Corrected HERE rather than at the conversion, because the verdict is recorded a few lines
    // down and a result that disagrees with its own origins is worse than either answer.
    const bool seasonOverrulesAbsolute = numbering == NumberingKind::Absolute && combinedMarkerSeen
                                         && info.season && info.episode.value_or(0) > 0;
    if (seasonOverrulesAbsolute) {
        numbering = NumberingKind::SeasonEpisode;
        info.numbering = numbering;
    }

    if (info.medium != MediumKind::Unknown)
        builder.verdict(Field::Medium, {std::string(label(info.medium)), analysis.contentKind.confidence});
    builder.verdict(Field::ContentKind, analysis.contentKind);
    builder.verdict(Field::AdultKind, analysis.adultKind);
    // The corrected answer, with the model's own confidence: the caller reading origins sees what
    // the result says. `season_episode` is the schema's spelling, the same one the labels use.
    builder.verdict(Field::NumberingKind,
                    seasonOverrulesAbsolute
                        ? SegmentedField{"season_episode", analysis.numberingKind.confidence}
                        : analysis.numberingKind);
    builder.verdict(Field::PackScope, analysis.packScope);
    builder.verdict(Field::SpecialKind, analysis.specialKind);

    switch (packScope) {
    case PackScope::Complete: info.explicitCompleteRange = true; [[fallthrough]];
    case PackScope::Season:
    case PackScope::MultiSeason:
    case PackScope::EpisodeBatch:
    case PackScope::Volume: info.pack = true; break;
    case PackScope::Single:
    case PackScope::Unknown: break;
    }
    switch (special) {
    case SpecialKind::Ova:
    case SpecialKind::Special: info.specials = true; break;
    case SpecialKind::Movie:
    case SpecialKind::None:
    case SpecialKind::Unknown: break;
    }
    if (numbering == NumberingKind::Absolute && info.episode.value_or(0) > 0) {
        info.absoluteEpisode = info.episode;
        info.absoluteEpisodeEnd = info.episodeEnd;
        info.episode.reset();
        info.episodeEnd.reset();
    }

    if (!info.editions.empty()) {
        std::ranges::sort(info.editions, {}, [](EditionKind value) {
            return static_cast<std::uint8_t>(value);
        });
        info.edition = info.editions.front();
    }
    info.valid = !info.title.empty();
    return info;
}

} // namespace neurelease::model
