// The closed vocabularies: one small pure function per field, raw span text in, canonical value
// out. `x265` -> HEVC, `BrRip` -> BluRay, `2160P` -> the P2160 tier, `DDP5.1.Atmos` -> codec +
// channels + feature. Aliases live in data/aliases/*.json and arrive here through the
// generated header; what remains hand-written is composition -- the rules for how compound tags
// split -- because that is logic, not data. A token no table knows converts to nothing and is
// reported unconverted, never rounded to the nearest value we happen to know.

#include "convert/field_values.hpp"
#include "convert/generated/aliases.hpp"

#include "text/regex.hpp"
#include "text/unicode.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace neurelease::convert {
namespace {

using text::Match;
using text::Regex;
using namespace std::string_view_literals;

int integer(std::string_view value) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} ? result : 0;
}

std::string stringOf(std::string_view value) { return std::string(value); }

bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.starts_with(prefix);
}

void eraseCharacters(std::string& value, std::string_view characters) {
    value.erase(std::remove_if(value.begin(), value.end(), [&](char c) {
        return characters.find(c) != std::string_view::npos;
    }), value.end());
}

std::string eraseMatches(std::string_view subject, const Regex& pattern) {
    std::string result;
    result.reserve(subject.size());
    std::size_t copied = 0;
    std::size_t from = 0;
    while (from <= subject.size()) {
        const Match match = pattern.matchAfter(subject, from);
        if (!match) break;
        const std::size_t begin = match.capturedStart();
        const std::size_t end = match.capturedEnd();
        result.append(subject.substr(copied, begin - copied));
        copied = end;
        from = end > begin ? end : end + 1;
    }
    result.append(subject.substr(copied));
    return result;
}

std::string replaceMatches(std::string_view subject, const Regex& pattern,
                           std::string_view replacement) {
    std::string result;
    result.reserve(subject.size());
    std::size_t copied = 0;
    std::size_t from = 0;
    while (from <= subject.size()) {
        const Match match = pattern.matchAfter(subject, from);
        if (!match) break;
        const std::size_t begin = match.capturedStart();
        const std::size_t end = match.capturedEnd();
        result.append(subject.substr(copied, begin - copied));
        result.append(replacement);
        copied = end;
        from = end > begin ? end : end + 1;
    }
    result.append(subject.substr(copied));
    return result;
}

struct Spelling {
    const char* pattern;
    const char* value;
};

struct CompiledSpelling {
    Regex pattern;
    std::string value;
};

template <std::size_t N>
std::vector<CompiledSpelling> compile(const Spelling (&spellings)[N]) {
    std::vector<CompiledSpelling> table;
    table.reserve(N);
    for (const Spelling& spelling : spellings)
        table.push_back({Regex(spelling.pattern, true), spelling.value});
    return table;
}

Reading firstMatch(std::string_view subject, const std::vector<CompiledSpelling>& table) {
    for (const CompiledSpelling& entry : table) {
        const Match match = entry.pattern.match(subject);
        if (!match) continue;
        return {entry.value, stringOf(match.captured(1)),
                static_cast<std::int32_t>(match.capturedStart(1)),
                static_cast<std::int32_t>(match.capturedEnd(1))};
    }
    return {};
}

struct StatedSize {
    int width = 0;
    int height = 0;
    std::string text;
    std::int32_t begin = -1;
    std::int32_t end = -1;
    bool exact = false;
    bool yieldsToExact = false;
};

bool isResolutionNumber(int pixels) {
    switch (pixels) {
    case 240: case 288: case 360: case 400: case 432: case 480: case 540: case 576:
    case 720: case 900: case 1080: case 1440: case 2160: case 4320:
    case 2560: case 3840: case 4096: case 7680:
        return true;
    default:
        return false;
    }
}

bool looksLikeWidth(int pixels) { return pixels >= 2560; }

template <typename Build>
void appendMatches(std::string_view subject, const Regex& pattern, Build build,
                   std::vector<StatedSize>& sizes) {
    std::size_t from = 0;
    while (from <= subject.size()) {
        const Match match = pattern.matchAfter(subject, from);
        if (!match) break;
        StatedSize size = build(match);
        if (size.width != 0 || size.height != 0) {
            size.text = text::simplified(match.captured());
            size.begin = static_cast<std::int32_t>(match.capturedStart());
            size.end = static_cast<std::int32_t>(match.capturedEnd());
            sizes.push_back(std::move(size));
        }
        const std::size_t begin = match.capturedStart();
        const std::size_t end = match.capturedEnd();
        from = end > begin ? end : end + 1;
    }
}

std::vector<StatedSize> statedSizes(std::string_view subject, bool acceptBareNumbers) {
    std::vector<StatedSize> sizes;
    static const Regex dimensions(R"((?<!\d)(\d{3,4})\s*[xX*]\s*(\d{3,4})(?!\d))");
    appendMatches(subject, dimensions, [](const Match& match) {
        return StatedSize{integer(match.captured(1)), integer(match.captured(2)), {}, -1, -1, true};
    }, sizes);

    static const Regex scanned(R"((?<!\d)(\d{3,4})[pi](?:\d{2,3})?\b)", true);
    appendMatches(subject, scanned, [](const Match& match) {
        const int pixels = integer(match.captured(1));
        return looksLikeWidth(pixels) ? StatedSize{pixels, 0, {}, -1, -1, true}
                                      : StatedSize{0, pixels, {}, -1, -1, true};
    }, sizes);

    static const Regex words(
        R"((?:^|[ ._\-/\[(,+~&;])(4K|UHD|8K|FHD|FullHD|Full[ ._-]HD|SD)(?:$|[ ._\-/\]),+~&;]))",
        true);
    appendMatches(subject, words, [](const Match& match) {
        std::string word = text::asciiUpper(match.captured(1));
        eraseCharacters(word, " .-_");
        if (word == "8K") return StatedSize{7680, 4320, {}, -1, -1, false, false};
        if (word == "UHD") return StatedSize{3840, 2160, {}, -1, -1, false, true};
        if (word == "4K") return StatedSize{3840, 2160, {}, -1, -1, false, false};
        if (word == "SD") return StatedSize{720, 480, {}, -1, -1, false, false};
        return StatedSize{1920, 1080, {}, -1, -1, false, false};
    }, sizes);

    static const Regex kNotation(
        R"((?:^|[ ._\-/\[(,+~&;])([235679])[ ._-]?K(?:$|[ ._\-/\]),+~&;]))", true);
    appendMatches(subject, kNotation, [](const Match& match) {
        return StatedSize{integer(match.captured(1)) * 1024, 0};
    }, sizes);

    if (!acceptBareNumbers) return sizes;
    static const Regex bare(R"((?<!\d)(\d{3,4})(?!\d))");
    std::vector<StatedSize> bareSizes;
    appendMatches(subject, bare, [](const Match& match) {
        const int pixels = integer(match.captured(1));
        if (!isResolutionNumber(pixels)) return StatedSize{};
        return looksLikeWidth(pixels) ? StatedSize{pixels, 0, {}, -1, -1, true}
                                      : StatedSize{0, pixels, {}, -1, -1, true};
    }, bareSizes);
    for (StatedSize& candidate : bareSizes) {
        const bool inside = std::ranges::any_of(sizes, [&](const StatedSize& size) {
            return candidate.begin >= size.begin && candidate.end <= size.end;
        });
        if (!inside) sizes.push_back(std::move(candidate));
    }
    return sizes;
}

const std::vector<CompiledSpelling>& hdrTable() {
    static const Spelling spellings[] = {
        {R"(\b(DV|DoVi|D\.V|Dolby[ ._-]?Vision)(?:[ ._-]?P[578])?\b)", "DV"},
        {R"(\b(HDR10\s*(?:\+|Plus|P\b))(?!\d))", "HDR10+"},
        {R"((?:^|[ ._\-/\[(,+~&;])(HLG)(?:$|[ ._\-/\]),+~&;]))", "HLG"},
        {R"(\b(HDR(?:10)?)\b)", "HDR10"},
        {R"(\b(SDR)\b)", "SDR"},
    };
    static const std::vector<CompiledSpelling> table = compile(spellings);
    return table;
}

HdrFormat formatOfLabel(std::string_view label) {
    if (label == "DV") return HdrFormat::DolbyVision;
    if (label == "HDR10+") return HdrFormat::Hdr10Plus;
    if (label == "HLG") return HdrFormat::Hlg;
    if (label == "HDR10") return HdrFormat::Hdr10;
    return HdrFormat::Sdr;
}

std::string cleanedSourceToken(std::string_view token) {
    std::string value = text::asciiUpper(token);
    eraseCharacters(value, "[](){} ._-");
    static const Regex gluedResolution(R"(\d{3,4}PI?$)", true);
    return eraseMatches(value, gluedResolution);
}

bool isDiscSpelling(std::string_view cleaned) {
    // BD25/BD50/BD66/BD100 name the disc capacity, BRMUX a Blu-ray remuxed into a container.
    // Sorted, because the lookup is a binary search.
    static constexpr std::array discs{
        "BD100"sv, "BD25"sv, "BD50"sv, "BD66"sv, "BDBOX"sv, "BDMUX"sv, "BLURAYRIP"sv, "BR"sv,
        "BRMUX"sv, "DEBD"sv, "DS4K"sv, "FRBD"sv, "ITBD"sv, "JPBD"sv, "UHD"sv, "UHDBD"sv,
        "UKBD"sv, "USBD"sv,
    };
    return std::ranges::binary_search(discs, cleaned);
}

} // namespace

std::string_view hdrFormatLabel(HdrFormat format) {
    switch (format) {
    case HdrFormat::Hlg: return "HLG";
    case HdrFormat::Hdr10: return "HDR10";
    case HdrFormat::Hdr10Plus: return "HDR10+";
    case HdrFormat::DolbyVision: return "DV";
    case HdrFormat::Sdr: return "SDR";
    }
    return {};
}

namespace {

struct ResolutionReading {
    Reading reading;
    int width = 0;
    int height = 0;
    bool exact = false;
};

std::string nearestResolutionTier(int width, int height) {
    const int impliedByWidth = width > 0 ? width * 9 / 16 : 0;
    const int effective = std::max(height, impliedByWidth);
    if (effective < 200) return {};
    static constexpr int tiers[] = {480, 720, 1080, 1440, 2160, 4320};
    int best = tiers[0];
    for (int tier : tiers)
        if (std::abs(tier - effective) < std::abs(best - effective)) best = tier;
    return std::to_string(best) + "p";
}

ResolutionReading resolutionDetailIn(std::string_view textValue, bool acceptBareNumbers) {
    std::string subject(textValue);
    if (acceptBareNumbers) {
        static const Regex letterOh(R"((?<=\d)[oO](?=[\dpi])|(?<=[oO])[oO](?=[\dpi]))");
        subject = replaceMatches(subject, letterOh, "0");
    }
    const std::vector<StatedSize> sizes = statedSizes(subject, acceptBareNumbers);
    const bool hasExact = std::ranges::any_of(sizes, [](const StatedSize& size) { return size.exact; });
    ResolutionReading best;
    int bestLines = 0;
    for (const StatedSize& size : sizes) {
        if (size.yieldsToExact && hasExact) continue;
        const std::string tier = nearestResolutionTier(size.width, size.height);
        if (tier.empty()) continue;
        const int lines = integer(std::string_view(tier).substr(0, tier.size() - 1));
        if (lines <= bestLines) continue;
        bestLines = lines;
        best = {{tier, size.text, size.begin, size.end}, size.width, size.height, size.exact};
    }
    return best;
}


} // namespace

ResolutionTier resolutionValue(std::string_view token) {
    const std::string& value = resolutionDetailIn(token, true).reading.value;
    if (value == "480p") return ResolutionTier::P480;
    if (value == "720p") return ResolutionTier::P720;
    if (value == "1080p") return ResolutionTier::P1080;
    if (value == "1440p") return ResolutionTier::P1440;
    if (value == "2160p") return ResolutionTier::P2160;
    if (value == "4320p") return ResolutionTier::P4320;
    return ResolutionTier::Unknown;
}

bool sourceTokenIsRemux(std::string_view token) { return contains(cleanedSourceToken(token), "REMUX"); }

bool sourceTokenIsLightEncode(std::string_view token) {
    const std::string value = cleanedSourceToken(token);
    return value == "MICROHD" || value == "HDLIGHT" || value == "MHD";
}

bool sourceTokenIsBareUhd(std::string_view token) {
    const std::string value = cleanedSourceToken(token);
    return value == "UHD" || value == "4K";
}

SourceKind sourceValue(std::string_view token) {
    const std::string value = cleanedSourceToken(token);
    if (value == "TS" || value == "TC" || contains(value, "TELESYNC") || contains(value, "TELECINE")
        || contains(value, "HDTS") || contains(value, "HDTC") || contains(value, "CAMRIP")
        || contains(value, "HDCAM") || value == "CAM") return SourceKind::Cam;
    if (contains(value, "WEB")) return contains(value, "DL") ? SourceKind::WebDl : SourceKind::WebRip;
    // DLMUX is an Italian-scene spelling for a web download remuxed into a container: the `DL`
    // is the source and the `MUX` the packaging, so it belongs with WEB-DL rather than with the
    // disc muxes above.
    if (value == "DLMUX") return SourceKind::WebDl;
    if (contains(value, "HDTV") || contains(value, "TVRIP") || contains(value, "PDTV")
        || contains(value, "SATRIP") || contains(value, "DVBRIP")
        || value == "SDTV" || value == "DVB") return SourceKind::Hdtv;
    if (contains(value, "DVD")) return SourceKind::Dvd;
    if (contains(value, "HDRIP")) return SourceKind::WebRip;
    if (contains(value, "REMUX") || contains(value, "BDMV") || value == "DISC"
        || isDiscSpelling(value)) return SourceKind::BluRay;
    if (contains(value, "BLURAY") || contains(value, "BDRIP") || contains(value, "BRRIP")
        || value == "BD" || value == "BURAY" || value.ends_with("SD-BD")) return SourceKind::BluRay;
    return SourceKind::Unknown;
}

VideoCodec codecValue(std::string_view token) {
    const std::string value = text::asciiUpper(token);
    if (contains(value, "AV1")) return VideoCodec::Av1;
    if (contains(value, "265") || contains(value, "HEVC")) return VideoCodec::Hevc;
    if (contains(value, "262") || contains(value, "MPG2")) return VideoCodec::Mpeg2;
    if (contains(value, "264") || contains(value, "AVC")) return VideoCodec::H264;
    if (contains(value, "XVID") || contains(value, "DIVX")) return VideoCodec::Xvid;
    if (contains(value, "MPEG2") || contains(value, "MPEG-2")) return VideoCodec::Mpeg2;
    if (contains(value, "VP9")) return VideoCodec::Vp9;
    if (contains(value, "VC-1") || contains(value, "VC1")) return VideoCodec::Vc1;
    if (startsWith(value, "HI10") || startsWith(value, "HI444")) return VideoCodec::H264;
    if (startsWith(value, "MAIN10") || startsWith(value, "MA10")) return VideoCodec::Hevc;
    if (contains(value, "WMV")) return VideoCodec::Wmv;
    if (contains(value, "VVC") || contains(value, "H266") || contains(value, "H.266")) return VideoCodec::Vvc;
    if (contains(value, "VP8")) return VideoCodec::Vp8;
    if (contains(value, "MPEG4") || contains(value, "MPEG-4")) return VideoCodec::Xvid;
    if (contains(value, "MPEG")) return VideoCodec::Mpeg2;
    return VideoCodec::Unknown;
}

static HdrReading hdrIn(std::string_view subject) {
    const Reading reading = firstMatch(subject, hdrTable());
    return {formatOfLabel(reading.value), reading};
}

HdrReading hdrValue(std::string_view token) {
    const HdrReading strict = hdrIn(token);
    if (strict.format != HdrFormat::Sdr || strict.reading.found()) return strict;
    const std::string value = text::asciiUpper(token);
    const auto reading = [token](std::string label) {
        return Reading{std::move(label), stringOf(token), 0, static_cast<std::int32_t>(token.size())};
    };
    if (contains(value, "DV") && contains(value, "HDR"))
        return {HdrFormat::DolbyVision, reading("DV")};
    if (contains(value, "HDR10+") || contains(value, "HDR10P"))
        return {HdrFormat::Hdr10Plus, reading("HDR10+")};
    if (contains(value, "HDR")) return {HdrFormat::Hdr10, reading("HDR10")};
    return {};
}

Reading bitDepthIn(std::string_view subject) {
    static const Spelling spellings[] = {
        {R"(\b(10[ ._-]?bits?)\b)", "10bit"},
        {R"(\b(10b)\b)", "10bit"},
        {R"(\b(Hi10P?)\b)", "10bit"},
        // GLUED TO A CODEC NAME: `Hevc10`, `x26510`, `AVC10`. The model gets one label per
        // segment and a codec token can state two facts, so it ranked `Hevc10` codec 0.79
        // against bit depth 0.18 and the depth was lost. Not a bare `10`, which is read
        // below only when the whole span is that number.
        {R"((?:HEVC|AVC|VP9|AV1|x26[45]|h[ ._-]?26[45])[ ._-]?(10)\b)", "10bit"},
        {R"(\b(Ma(?:in)?10P?)\b)", "10bit"},
        {R"(\b(YUV\d{3}P10)\b)", "10bit"},
        {R"(\b(8[ ._-]?bits?)\b)", "8bit"},
        {R"(\b(12[ ._-]?bits?)\b)", "12bit"},
    };
    static const std::vector<CompiledSpelling> table = compile(spellings);
    const Reading reading = firstMatch(subject, table);
    if (reading.found()) return reading;
    static const Regex bare(R"(^\s*(8|10|12)\s*$)");
    const Match match = bare.match(subject);
    if (!match) return {};
    return {stringOf(match.captured(1)) + "bit", stringOf(match.captured(1)),
            static_cast<std::int32_t>(match.capturedStart(1)),
            static_cast<std::int32_t>(match.capturedEnd(1))};
}

namespace {

const std::vector<CompiledSpelling>& editionSpellings() {
    static const Spelling spellings[] = {
        {R"((?:^|[ ._\-\[(])(Director.?s[ ._-]?Cut|DC(?=$|[ ._-]))(?:$|[^A-Za-z]))", "Director's Cut"},
        {R"((?:^|[ ._\-\[(])(Final[ ._-]?Cut)(?:$|[^A-Za-z]))", "Final Cut"},
        {R"((?:^|[ ._\-\[(])(Extended(?:[ ._-]?(?:Cut|Edition|Version))?)(?:$|[^A-Za-z]))", "Extended"},
        {R"((?:^|[ ._\-\[(])(IMAX(?:[ ._-]?Enhanced)?)(?:$|[^A-Za-z]))", "IMAX"},
        {R"((?:^|[ ._\-\[(])(Redux)(?:$|[^A-Za-z]))", "Redux"},
        {R"((?:^|[ ._\-\[(])(Theatrical(?:[ ._-]?Cut)?)(?:$|[^A-Za-z]))", "Theatrical"},
        {R"((?:^|[ ._\-\[(])(Uncut)(?:$|[^A-Za-z]))", "Uncut"},
        {R"((?:^|[ ._\-\[(])(Unrated)(?:$|[^A-Za-z]))", "Unrated"},
        {R"((?:^|[ ._\-\[(])(Remaster(?:ed)?)(?:$|[^A-Za-z]))", "Remastered"},
        {R"((?:^|[ ._\-\[(])(Criterion(?:[ ._-]?Collection)?)(?:$|[^A-Za-z]))", "Criterion"},
        {R"((?:^|[ ._\-\[(])(Open[ ._-]?Matte)(?:$|[^A-Za-z]))", "Open Matte"},
        {R"((?:^|[ ._\-\[(])(Uncensored)(?:$|[^A-Za-z]))", "Uncensored"},
        {R"((?:^|[ ._\-\[(])(Special[ ._-]?Edition|SE(?=[ ._-]))(?:$|[^A-Za-z]))", "Special Edition"},
        {R"((?:^|[ ._\-\[(])(Deluxe(?:[ ._-]?Edition)?)(?:$|[^A-Za-z]))", "Deluxe"},
        // APPENDED BELOW THE FOURTEEN ABOVE, because the table is read in order and the first
        // entry that matches becomes the PRIMARY edition. These eight are rarer and weaker
        // identifiers than the originals, so a `Criterion 40th Anniversary Edition` stays
        // Criterion; they still all land in `editions`.
        {R"((?:^|[ ._\-\[(])(Despecialized)(?:$|[^A-Za-z]))", "Despecialized"},
        {R"((?:^|[ ._\-\[(])(Assembly[ ._-]?Cut)(?:$|[^A-Za-z]))", "Assembly Cut"},
        // THE ORDINAL IS OPTIONAL BUT `Edition` IS NOT. `25th Anniversary Edition` and
        // `Anniversary Edition` are both editions; a bare `Anniversary` is an ordinary word that
        // belongs to plenty of titles (`Anniversary.2023.1080p`), so it is not taken alone.
        {R"((?:^|[ ._\-\[(])((?:\d{1,3}(?:th|st|nd|rd)[ ._-]?)?Anniversary[ ._-]?Edition)(?:$|[^A-Za-z]))",
         "Anniversary"},
        {R"((?:^|[ ._\-\[(])(Signature[ ._-]?Edition)(?:$|[^A-Za-z]))", "Signature"},
        {R"((?:^|[ ._\-\[(])(Imperial[ ._-]?Edition)(?:$|[^A-Za-z]))", "Imperial"},
        {R"((?:^|[ ._\-\[(])(Diamond[ ._-]?Edition)(?:$|[^A-Za-z]))", "Diamond"},
        // TWO EPISODES IN ONE FILE. It ENDS in a digit, so the usual letters-only trailing guard
        // would let `2in1080p` through; this one refuses a following digit as well.
        {R"((?:^|[ ._\-\[(])(2[ ._-]?in[ ._-]?1)(?:$|[^A-Za-z0-9]))", "2in1"},
        {R"((?:^|[ ._\-\[(])(Pre[ ._-]?Air)(?:$|[^A-Za-z]))", "Preair"},
    };
    static const std::vector<CompiledSpelling> table = compile(spellings);
    return table;
}

EditionKind editionOfLabel(std::string_view value) {
    if (value == "IMAX") return EditionKind::Imax;
    if (value == "Criterion") return EditionKind::Criterion;
    if (value == "Open Matte") return EditionKind::OpenMatte;
    if (value == "Remastered") return EditionKind::Remastered;
    if (value == "Unrated") return EditionKind::Unrated;
    if (value == "Uncut") return EditionKind::Uncut;
    if (value == "Uncensored") return EditionKind::Uncensored;
    if (value == "Special Edition") return EditionKind::SpecialEdition;
    if (value == "Deluxe") return EditionKind::Deluxe;
    if (value == "Redux") return EditionKind::Redux;
    if (value == "Extended") return EditionKind::Extended;
    if (value == "Director's Cut") return EditionKind::DirectorsCut;
    if (value == "Final Cut") return EditionKind::FinalCut;
    if (value == "Theatrical") return EditionKind::Theatrical;
    if (value == "Despecialized") return EditionKind::Despecialized;
    if (value == "Assembly Cut") return EditionKind::AssemblyCut;
    if (value == "Anniversary") return EditionKind::Anniversary;
    if (value == "Signature") return EditionKind::Signature;
    if (value == "Imperial") return EditionKind::Imperial;
    if (value == "Diamond") return EditionKind::Diamond;
    if (value == "2in1") return EditionKind::TwoInOne;
    if (value == "Preair") return EditionKind::Preair;
    return EditionKind::Unknown;
}

} // namespace

EditionKind editionIn(std::string_view subject) {
    return editionOfLabel(firstMatch(subject, editionSpellings()).value);
}

std::vector<EditionKind> editionsIn(std::string_view subject) {
    std::vector<EditionKind> found;
    for (const CompiledSpelling& spelling : editionSpellings()) {
        if (!spelling.pattern.match(subject)) continue;
        const EditionKind edition = editionOfLabel(spelling.value);
        if (edition != EditionKind::Unknown && std::ranges::find(found, edition) == found.end())
            found.push_back(edition);
    }
    std::ranges::sort(found, {}, [](EditionKind value) { return static_cast<std::uint8_t>(value); });
    return found;
}

std::string audioProfileValue(std::string_view token) {
    const std::string upper = text::asciiUpper(text::trimmed(token));
    std::string squeezed;
    for (const char character : upper)
        if (character != ' ' && character != '.' && character != '_' && character != '-'
            && character != ':')
            squeezed.push_back(character);
    if (squeezed.find("ATMOS") != std::string::npos) return "Atmos";
    if (squeezed.find("DTSX") != std::string::npos) return "DTS:X";
    if (squeezed.find("AURO3D") != std::string::npos || squeezed == "AURO") return "Auro-3D";
    return {};
}

std::string audioCodecValue(std::string_view token) {
    static const Regex layout(R"([ ._-]?(?:2[ ._]?0|5[ ._]?1|7[ ._]?1|6[ ._]?CH|\d[ ._]?\dch|ch)\s*$)", true);
    static const Regex leadingMultiplier(R"(^\d+[ ._-]?X[ ._-]?)", true);
    static const Regex trailingMultiplier(R"([ ._-]?X[ ._-]?\d+$)", true);
    std::string value = text::asciiUpper(text::trimmed(token));
    const bool atmos = contains(value, "ATMOS");
    if (atmos) {
        std::size_t at;
        while ((at = value.find("ATMOS")) != std::string::npos) value.erase(at, 5);
        value = stringOf(text::trimmed(value));
    }
    value = eraseMatches(value, layout);
    value = eraseMatches(value, leadingMultiplier);
    value = eraseMatches(value, trailingMultiplier);
    eraseCharacters(value, " _-.");
    if (value.empty()) return atmos ? "Atmos" : std::string{};
    if (contains(value, "TRUEHD")) return "TrueHD";
    if (contains(value, "DTSHDMA") || contains(value, "DTSHD") || contains(value, "DTSMA")
        || contains(value, "DTSHR") || contains(value, "DTSMASTER")) return "DTS-HD MA";
    if (contains(value, "DTSX")) return "DTS:X";
    if (contains(value, "DTS")) return "DTS";
    if (contains(value, "EAC3") || contains(value, "DDP") || contains(value, "DD+")
        || contains(value, "DDPA") || contains(value, "DOLBYDIGITALPLUS")
        || contains(value, "DOLBYDPLUS")) return "DDP";
    if (contains(value, "AC3") || contains(value, "DOLBYD") || value == "DD") return "DD";
    if (contains(value, "AAC")) return "AAC";
    if (contains(value, "FLAC")) return "FLAC";
    if (contains(value, "OPUS")) return "OPUS";
    if (contains(value, "VORBIS")) return "VORBIS";
    if (contains(value, "ALAC")) return "ALAC";
    if (contains(value, "PCM")) return "PCM";
    if (contains(value, "MP3")) return "MP3";
    if (value.size() > 2 && value.back() >= '0' && value.back() <= '9')
        return audioCodecValue(std::string_view(value).substr(0, value.size() - 1));
    return atmos ? "Atmos" : std::string{};
}

std::string audioChannelsValue(std::string_view token) {
    std::string channels = text::asciiUpper(token);
    std::ranges::replace(channels, ' ', '.');
    std::ranges::replace(channels, '_', '.');
    if (channels.ends_with("CH")) channels.erase(channels.size() - 2);
    while (!channels.empty() && channels.back() == '.') channels.pop_back();
    if (channels == "8") return "7.1";
    if (channels == "6") return "5.1";
    if (channels == "2") return "2.0";
    // A COUNT WRITTEN AS A LAYOUT IS STILL A COUNT. `X.Y` names X full-range channels plus Y LFE,
    // so 5.1 IS six channels - and a tool that read the channel count and formatted it `6.0`
    // states the same track. A literal 6.0 (six full-range, no subwoofer) is not a layout consumer
    // releases ship, and `EAC3 6.0` on an Amazon WEB-DL is 5.1. This only makes the two spellings
    // agree: `6CH` already answered 5.1 through the branch above while `6.0` fell through
    // unchanged, so the same fact got two answers depending on how the ripper wrote it.
    // 22 names in the 60k label corpus say `6.0` and 228 say `6CH`; 7 say `8.0` and 34 `8CH`.
    // Nothing else is folded: 2.0 is genuinely stereo (1,233 names), and 3.0, 4.0, 5.0 and 7.0
    // are rare enough and ambiguous enough that guessing at them would cost more than it buys.
    if (channels == "6.0") return "5.1";
    if (channels == "8.0") return "7.1";
    return channels;
}

MarkerNumbers markerNumbersIn(std::string_view subject) {
    static const Regex numbers(R"((\d{1,4}))");
    std::vector<int> found;
    // A season written as a word - `Second Season`, `Fourth Stage`, `Season III` - states one
    // number as plainly as `S02`; the model marks the span, this reads it.
    {
        static const std::pair<const char*, int> words[] = {
            {"FIRST", 1}, {"SECOND", 2}, {"THIRD", 3}, {"FOURTH", 4}, {"FIFTH", 5}, {"SIXTH", 6},
            {"SEVENTH", 7}, {"EIGHTH", 8}, {"NINTH", 9}, {"TENTH", 10}, {"1ST", 1}, {"2ND", 2},
            {"3RD", 3}, {"4TH", 4}, {"5TH", 5}, {"6TH", 6}, {"7TH", 7}, {"8TH", 8}, {"9TH", 9}};
        static const std::pair<const char*, int> romans[] = {
            {"III", 3}, {"II", 2}, {"IV", 4}, {"VI", 6}, {"VII", 7}, {"VIII", 8}, {"IX", 9}, {"V", 5}, {"I", 1}};
        const std::string upper = text::asciiUpper(subject);
        const bool hasDigit = std::any_of(upper.begin(), upper.end(),
                                          [](char c) { return c >= '0' && c <= '9'; });
        if (!hasDigit) {
            for (const auto& [word, number] : words)
                if (upper.find(word) != std::string::npos) return {number, 0, 1, true};
            static const Regex trailingRoman(R"((?:^|[^A-Z])(X{0,1}(?:IX|IV|V?I{0,3}))$)");
            if (const Match roman = trailingRoman.match(upper); roman && !roman.captured(1).empty())
                for (const auto& [numeral, number] : romans)
                    if (roman.captured(1) == numeral) return {number, 0, 1, true};
        }
    }
    std::size_t from = 0;
    while (from <= subject.size()) {
        const Match match = numbers.matchAfter(subject, from);
        if (!match) break;
        found.push_back(integer(match.captured(1)));
        from = match.capturedEnd() > match.capturedStart() ? match.capturedEnd()
                                                           : match.capturedEnd() + 1;
    }
    if (found.empty()) return {};
    MarkerNumbers result{found.front(), 0, 1, true};
    for (std::size_t at = 1; at < found.size(); ++at) {
        const int value = found[at];
        if (value <= (result.last > 0 ? result.last : result.first)) break;
        result.last = value;
        ++result.count;
    }
    return result;
}

EpisodeMarkerReading episodeMarkerIn(std::string_view subject) {
    // THE CROSS AND POSITION NOTATIONS. `2x11` states season 2 episode 11 and `4of10` episode 4 of
    // a ten-episode run; read as bare numbers by markerNumbersIn both become the false ranges 2-11
    // and 4-10. These two lived inline in the mapping walk; they are here so the mapper reads one
    // named function and the notation rules sit beside the numbers they reinterpret.
    //
    // TRIED AND REMOVED, so it is not re-proposed: extending the cross form to a chained run
    // (`1x02x03x04`) and to a four-digit left operand (`1940x01`). Both are real notations and
    // both bought three fixture names on GuessIt's corpus, but across 3,317 real names the chained
    // form occurs once and the wide form never - and on that one name the rule produced a season
    // the gold does not want. A rule whose only real-world appearance is a regression is a rule
    // that was fitted to a test corpus.
    // THE EPISODE WORD MAY LEAD, AND THE CROSS MAY BE THE MULTIPLICATION SIGN. `Ep 2x03` read as
    // episode 2 and `2×7` as episode 2 with no season, both on GuessIt's corpus (2026-09-16), each
    // a span the model typed correctly. The left operand stays two digits: `1920x1080` is a frame.
    static const Regex cross(R"(^(?:ep(?:isode)?[ ._-]?)?(\d{1,2})[ ._-]?[x×][ ._-]?(\d{1,4})$)", true);
    // `S8E6` AND `T01E08` AS ONE EPISODE SPAN. The model separates the season letter from the
    // episode when it can; when it hands both over in one span - the Spanish `T` for temporada, or
    // an `S` it did not cut - the numbers reader answered episode 8 for season 8 episode 6, and
    // `T01XE08` (the cross written between them) episode 1. The letter says which number is which.
    static const Regex seasonEpisode(R"(^[st](\d{1,2})[ ._-]?x?[ ._-]?e(\d{1,4})$)", true);
    static const Regex positionOf(R"(^(\d{1,4})[ ._-]?of[ ._-]?(\d{1,4})$)", true);
    if (const Match match = cross.match(subject))
        return {integer(match.captured(1)), integer(match.captured(2)), 0, 1, true};
    if (const Match match = seasonEpisode.match(subject))
        return {integer(match.captured(1)), integer(match.captured(2)), 0, 1, true};
    if (const Match match = positionOf.match(subject))
        return {0, integer(match.captured(1)), 0, 1, true};
    const MarkerNumbers numbers = markerNumbersIn(subject);
    return {0, numbers.first, numbers.last, numbers.count, numbers.stated};
}

EpisodeMarkerReading seasonEpisodeMarkerIn(std::string_view subject) {
    // Every digit in the span, in order. `Cap.104` -> "104"; `Cap.101_108` -> "101108", a range.
    std::string digits;
    for (char c : subject)
        if (c >= '0' && c <= '9') digits.push_back(c);
    // A RANGE IS TWO OF THEM SIDE BY SIDE, sharing one season: `Cap.101_108` is season 1,
    // episodes 1 to 8. Six digits with the same leading half is that and nothing else.
    if (digits.size() >= 6 && digits.size() % 2 == 0) {
        const std::size_t half = digits.size() / 2;
        const std::string_view first(digits.data(), half);
        const std::string_view second(digits.data() + half, half);
        if (first.substr(0, half - 2) == second.substr(0, half - 2)) {
            const int season = integer(first.substr(0, half - 2));
            const int from = integer(first.substr(half - 2));
            const int to = integer(second.substr(half - 2));
            if (to > from) return {season, from, to, to - from + 1, true};
            return {season, from, 0, 1, true};
        }
    }
    if (digits.size() < 3) {
        const MarkerNumbers numbers = markerNumbersIn(subject);
        return {0, numbers.first, numbers.last, numbers.count, numbers.stated};
    }
    return {integer(std::string_view(digits).substr(0, digits.size() - 2)),
            integer(std::string_view(digits).substr(digits.size() - 2)), 0, 1, true};
}

MediumKind mediumOfContainer(std::string_view container) {
    const std::string lowered = text::asciiLower(container);
    const auto found = std::ranges::lower_bound(
        generated::containers, lowered, {}, &generated::Alias::input);
    if (found == generated::containers.end() || found->input != lowered) return MediumKind::Unknown;
    if (found->output == "video") return MediumKind::Video;
    if (found->output == "music") return MediumKind::Music;
    if (found->output == "audiobook") return MediumKind::Audiobook;
    if (found->output == "book") return MediumKind::Book;
    if (found->output == "comic") return MediumKind::Comic;
    if (found->output == "software") return MediumKind::Software;
    if (found->output == "game") return MediumKind::Game;
    if (found->output == "image") return MediumKind::Image;
    if (found->output == "subtitle") return MediumKind::Subtitle;
    if (found->output == "archive") return MediumKind::Archive;
    return MediumKind::Unknown;
}

bool isChecksum(std::string_view text) {
    std::size_t digits = 0;
    for (const char c : text) {
        if (c == '[' || c == ']' || c == '(' || c == ')' || c == '{' || c == '}'
            || c == ' ' || c == '_' || c == '-' || c == '.')
            continue;
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!hex) return false;
        ++digits;
    }
    return digits == 8;
}

std::string containerValue(std::string_view token) {
    std::string value = text::asciiLower(text::trimmed(token));
    if (const std::size_t dot = value.rfind('.'); dot != std::string::npos) value.erase(0, dot + 1);
    static const Regex alphanumeric(R"(^[a-z0-9]{1,9}$)");
    if (!alphanumeric.match(value) || mediumOfContainer(value) == MediumKind::Unknown) return {};
    return value;
}

bool statesAudioSampleDepth(std::string_view token) {
    static const Regex pattern(
        R"(\b(?:16|24|32)[ ._-]?(?:bits?|BIT)\b|\b(?:16|24)[ ._-](?:44|48|88|96|176|192)\b|\b(?:16|24|32)[ ._-]?bits?[ ._-]?(?:\d+(?:\.\d+)?[ ._-]?k?hz)?\b)",
        true);
    return static_cast<bool>(pattern.match(token));
}

bool statesSoftwareBitness(std::string_view token) {
    static const Regex pattern(R"(\b(?:32|64)[ ._-]?bits?\b|\bx(?:86|64)[ ._-]?(?:32|64)?\b)", true);
    return static_cast<bool>(pattern.match(token));
}

namespace {

struct LanguageMatcher {
    Regex pattern;
    std::string code;
};

const std::vector<LanguageMatcher>& compiledLanguageTags() {
    static const std::vector<LanguageMatcher> matchers = [] {
        std::vector<LanguageMatcher> result;
        for (const generated::LanguageAlias& tag : generated::languages) {
            std::string expression = "(?:^|[ ._\\-/\\[(,+~&;|])(?:";
            expression += tag.expression;
            expression += ")(?:$|[ ._\\-/\\]),+~&;|])";
            result.push_back({Regex(expression, true), std::string(tag.code)});
        }
        return result;
    }();
    return matchers;
}

void appendCodes(std::string_view subject, std::vector<std::string>& codes) {
    for (const LanguageMatcher& entry : compiledLanguageTags()) {
        if (!entry.pattern.match(subject)) continue;
        if (std::ranges::find(codes, entry.code) == codes.end()) codes.push_back(entry.code);
    }
}

std::string withoutSubtitleDecoration(std::string_view token) {
    static const Regex decoration(
        R"((?:^\d+[ ._-]?x[ ._-]?)|(?:[ ._-]?x\d+$)|SOFT|HARD|FORCED|FULL|INC|MTL|SUBBED|SUBTITLES?|SUBS?|TITULKY|TIT|NAPISY|LEGENDA(?:DO|S)?|DUB(?:BED)?|AUDIO|SRT|ASS|SSA|SUP|PGS|IDX|VOB)",
        true);
    return stringOf(text::trimmed(eraseMatches(token, decoration)));
}

bool tokenSeparator(char c) {
    constexpr std::string_view separators = " ._-+|&/,()[]";
    return separators.find(c) != std::string_view::npos;
}

std::vector<std::string> tokenParts(std::string_view token) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    const auto appendPart = [&](std::size_t first, std::size_t last, auto& output) {
        if (last > first) output.emplace_back(token.substr(first, last - first));
    };
    for (std::size_t at = 0; at <= token.size(); ++at) {
        if (at == token.size() || tokenSeparator(token[at])) {
            appendPart(begin, at, parts);
            begin = at + 1;
        }
    }
    const std::vector<std::string> unsplit = parts;
    for (const std::string& part : unsplit) {
        std::size_t hump = 0;
        for (std::size_t at = 1; at < part.size(); ++at) {
            const unsigned char before = static_cast<unsigned char>(part[at - 1]);
            const unsigned char current = static_cast<unsigned char>(part[at]);
            if (before >= 'a' && before <= 'z' && current >= 'A' && current <= 'Z') {
                if (at > hump) parts.emplace_back(part.substr(hump, at - hump));
                hump = at;
            }
        }
        if (hump > 0 && hump < part.size()) parts.emplace_back(part.substr(hump));
    }
    return parts;
}

} // namespace

std::vector<std::string> languageCodesOfToken(std::string_view token) {
    std::vector<std::string> codes;
    appendCodes(token, codes);
    if (!codes.empty()) return codes;
    const std::string stripped = withoutSubtitleDecoration(token);
    if (!stripped.empty() && stripped != token) appendCodes(stripped, codes);
    if (!codes.empty()) return codes;
    for (const std::string& part : tokenParts(token)) {
        appendCodes(part, codes);
        const std::string bare = withoutSubtitleDecoration(part);
        if (!bare.empty() && bare != part) appendCodes(bare, codes);
    }
    return codes;
}

SubtitleFormat subtitleFormatValue(std::string_view token) {
    const std::string value = text::asciiUpper(token);
    if (contains(value, "VOBSUB") || contains(value, "IDX")) return SubtitleFormat::VobSub;
    if (contains(value, "PGS") || contains(value, "SUP")) return SubtitleFormat::Pgs;
    if (contains(value, "SRT")) return SubtitleFormat::Srt;
    if (contains(value, "VTT")) return SubtitleFormat::Vtt;
    if (contains(value, "SMI")) return SubtitleFormat::Smi;
    if (contains(value, "SSA") || contains(value, "ASS")) return SubtitleFormat::Ass;
    return SubtitleFormat::Unknown;
}

bool statesSubtitlesOnly(std::string_view token) {
    static const Regex pattern(
        R"(^\W*(?:\d+[ ._-]?x[ ._-]?)?(?:ASS|SSA|SRT|SUP|PGS|IDX|VOB[ ._-]?SUB|DVD[ ._-]?SUB|D[ ._-]?SUB|CC|SOFT[ ._-]?SUBS?|HARD[ ._-]?SUBS?|SOFT|SUBBED|SUBTITLES?|SUBS?|W[ ._-]?SUBS?|INC[ ._-]?SUBS?)(?:[ ._-]?x?\d+)?\W*$)",
        true);
    return static_cast<bool>(pattern.match(text::trimmed(token)));
}

bool statesSeveralSubtitles(std::string_view token) {
    static const Regex pattern(
        R"(\b(?:MULTI?[ ._-]?(?:SUBS?|SUBTITLES?|SUBBED)|MULTIPLE[ ._-]?SUBTITLES?|DUAL[ ._-]?SUBS?|MORE[ ._-]?SUBTITLES?|MSUBS?|MULTI\d)\b)",
        true);
    if (pattern.match(token)) return true;
    static const Regex counted(R"(\b(?:ASS|SSA|SRT|SUP|SUB)[ ._-]?x[ ._-]?[2-9]\b)", true);
    return static_cast<bool>(counted.match(token));
}

namespace {

bool validDate(int year, int month, int day) {
    if (year < 1 || month < 1 || month > 12 || day < 1) return false;
    static constexpr int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int maximum = days[month - 1];
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    if (month == 2 && leap) maximum = 29;
    return day <= maximum;
}

Date makeDate(int year, int month, int day) {
    if (!validDate(year, month, day)) return {};
    return {static_cast<std::int16_t>(year), static_cast<std::int8_t>(month),
            static_cast<std::int8_t>(day)};
}

std::string isoDate(const Date& date) {
    std::array<char, 11> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%04d-%02d-%02d", date.year,
                  static_cast<int>(date.month), static_cast<int>(date.day));
    return buffer.data();
}

DateReading dateReading(const Match& match, int year, int month, int day) {
    const Date date = makeDate(year, month, day);
    if (!date.valid()) return {};
    return {date, {isoDate(date), stringOf(match.captured()),
                   static_cast<std::int32_t>(match.capturedStart()),
                   static_cast<std::int32_t>(match.capturedEnd())}};
}

} // namespace

std::string platformValue(std::string_view token) {
    std::string cleaned = stringOf(text::trimmed(token));
    eraseCharacters(cleaned, " ");
    std::string bare = text::asciiUpper(cleaned);
    eraseCharacters(bare, "-.+_");
    const auto found = std::ranges::lower_bound(
        generated::platformAliases, bare, {}, &generated::Alias::input);
    if (found != generated::platformAliases.end() && found->input == bare)
        return stringOf(found->output);
    if (bare == "IP") return "iP";
    return text::asciiUpper(cleaned);
}

DateReading dateIn(std::string_view subject) {
    static const Regex yearFirst(R"((?<!\d)((?:19|20)\d{2})[ ._-]?(\d{2})[ ._-]?(\d{2})(?!\d))");
    for (std::size_t from = 0; from < subject.size();) {
        const Match match = yearFirst.match(subject, from);
        if (!match) break;
        DateReading result = dateReading(match, integer(match.captured(1)), integer(match.captured(2)),
                                         integer(match.captured(3)));
        if (result.date.valid()) return result;
        from = match.capturedStart(1) + 1;
    }

    static const Regex separated(R"((?<!\d)(\d{2})[ ._-](\d{2})[ ._-](\d{2})(?!\d))");
    for (std::size_t from = 0; from <= subject.size();) {
        const Match match = separated.matchAfter(subject, from);
        if (!match) break;
        const int year = integer(match.captured(1));
        if (year >= 6 && year <= 30) {
            DateReading result = dateReading(match, 2000 + year, integer(match.captured(2)),
                                             integer(match.captured(3)));
            if (result.date.valid()) return result;
        }
        from = match.capturedEnd() > match.capturedStart() ? match.capturedEnd()
                                                           : match.capturedEnd() + 1;
    }

    static const Regex compact(R"((?<!\d)(\d{2})(\d{2})(\d{2})(?!\d))");
    for (std::size_t from = 0; from < subject.size();) {
        const Match match = compact.match(subject, from);
        if (!match) break;
        const int shortYear = integer(match.captured(1));
        const int year = shortYear <= 79 ? 2000 + shortYear : 1900 + shortYear;
        DateReading result = dateReading(match, year, integer(match.captured(2)),
                                         integer(match.captured(3)));
        if (result.date.valid()) return result;
        from = match.capturedStart(1) + 1;
    }
    return {};
}

Reading yearIn(std::string_view subject, std::int32_t excludeBegin, std::int32_t excludeEnd) {
    static const Regex pattern(R"((?<![\dxX])((?:19|20)\d{2})(?![\d]|\s*[xX]\s*\d{3,4}))");
    Reading result;
    std::size_t from = 0;
    while (from <= subject.size()) {
        const Match match = pattern.matchAfter(subject, from);
        if (!match) break;
        const auto begin = static_cast<std::int32_t>(match.capturedStart(1));
        const auto end = static_cast<std::int32_t>(match.capturedEnd(1));
        if (!(excludeBegin >= 0 && begin >= excludeBegin && end <= excludeEnd))
            result = {stringOf(match.captured(1)), stringOf(match.captured(1)), begin, end};
        from = match.capturedEnd() > match.capturedStart() ? match.capturedEnd()
                                                           : match.capturedEnd() + 1;
    }
    return result;
}

} // namespace neurelease::convert
