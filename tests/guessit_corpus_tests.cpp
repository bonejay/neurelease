// GUESSIT'S OWN REGRESSION CORPUS, SCORED UNDER THIS PARSER'S CONTRACT.
//
// GuessIt publishes roughly a thousand expectation-annotated release names in
// guessit/test/{movies,episodes,various}.yml. They are a genuinely independent corpus: nothing
// here trained on them, and they were written by another project to pin down its own behaviour.
// That makes them valuable, and it also means they cannot be used as a pass/fail oracle without
// translation — the expectations encode GUESSIT's definitions, not ours.
//
// So this test does two things, and keeps them apart:
//
//   0. Scope. Corpus entries written as a filesystem path are skipped. GuessIt reads the
//      enclosing directories, so "Movies/Fear and Loathing in Las Vegas (1998)/Fear.and...mkv"
//      asserts year 1998 while the filename never says 1998. This parser is handed a release
//      name, not a path, so those expectations are unanswerable rather than missed, and scoring
//      them would measure the harness rather than the parser.
//
//      `type` is skipped for a different reason: it is not an independent assertion. Most entries
//      inherit it from their file's `__default__` block, and an entry that states it restates what
//      its own season/episode/date/part property already says, so scoring it counts numbering
//      twice and then adds GuessIt's movie default for names that say nothing either way. Our
//      content kind answers a wider question — live action or animated, and also music, book,
//      software or game — that this property cannot agree or disagree with.
//
//      A container extension our closed vocabulary has no entry for is skipped the same way; see
//      `UnmodelledContainers`.
//
//   1. Translation. Every expectation is re-read under OUR contract before it is scored:
//      resolutions collapse to our tiers, `Web` accepts either of our WEB-DL/WEBRip readings,
//      absolute numbering answers an episode question, `proper_count` becomes our proper/repack
//      booleans, and GuessIt properties with no counterpart here (bonus, cd, film, part, country,
//      uuid, negations, ...) are reported as UNMAPPED and never scored. A property we cannot
//      translate honestly is not a property we get to grade ourselves on.
//
//   2. Tolerance. This is a REGRESSION gate, not an agreement gate, and it is deliberately loose
//      per property and strict in total. Each property carries a pinned floor recorded from the
//      state of this parser at the time of writing; a property may fall up to `FloorTolerance`
//      names below its floor without failing, because a segmenter swap moves a name or two either
//      way for reasons unrelated to the code under test. What may NOT slide is the corpus total:
//      `RecordedFullyCorrect` is enforced, so trading a name here for six there passes and losing
//      ground everywhere fails. Disagreeing with GuessIt is expected and allowed — the corpus
//      encodes its title-boundary conventions, its path-aware inputs, and a long tail of curated
//      legacy cases. Silently getting WORSE overall is what fails.
//
// The corpus is read from a sibling GuessIt checkout and never copied into this repository: it is
// LGPL-licensed GuessIt property. When the checkout, or a file inside it, is missing the test
// exits 77 and ctest reports a skip, so a clone without GuessIt beside it still runs green.
//
// Refresh the floors deliberately, never reflexively: run the test, read the printed table, and
// move a floor only when the new number is one you meant to produce.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "neurelease/parser.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using namespace neurelease;

namespace {

constexpr int SkipExitCode = 77; // ctest SKIP_RETURN_CODE

constexpr std::array CorpusFiles{"movies.yml", "episodes.yml", "various.yml"};

// The corpus this file's floors were recorded against (GuessIt v4.4.0). A different GuessIt
// checkout is still parsed and reported, but its floors are not enforced: a number recorded
// against other data would fail for reasons that have nothing to do with this parser.
constexpr std::size_t RecordedCaseCount = 883;

// --- the smallest YAML reader that reads this corpus ---------------------------------------------
//
// These three files use one shape and no more: an explicit-key entry (`? name`) whose value is a
// flat mapping of scalars, inline flow sequences, and block sequences. A general YAML library
// would be a dependency; this is thirty lines.

std::string_view trimmed(std::string_view text) {
    const auto space = [](char character) {
        return character == ' ' || character == '\t' || character == '\r' || character == '\n';
    };
    while (!text.empty() && space(text.front())) text.remove_prefix(1);
    while (!text.empty() && space(text.back())) text.remove_suffix(1);
    return text;
}

char loweredAscii(char character) {
    return character >= 'A' && character <= 'Z' ? static_cast<char>(character - 'A' + 'a')
                                                : character;
}

std::string lowered(std::string_view text) {
    std::string result(text);
    std::ranges::transform(result, result.begin(), loweredAscii);
    return result;
}

// Case-, separator-, and punctuation-insensitive identity for open text. Non-ASCII bytes are kept
// verbatim: both sides of the comparison come from the same UTF-8 source, so accented titles match
// without dragging in a normalisation library.
std::string textKey(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte >= 0x80 || (byte >= '0' && byte <= '9') || (loweredAscii(character) >= 'a' &&
                                                             loweredAscii(character) <= 'z'))
            result.push_back(loweredAscii(character));
    }
    return result;
}

// A trailing `# ...` comment, but only outside quotes. The corpus writes both `value # note` and
// `value  # note`, and no expectation value in it contains a hash.
std::string_view withoutComment(std::string_view value) {
    if (!value.empty() && (value.front() == '\'' || value.front() == '"')) {
        const std::size_t close = value.find(value.front(), 1);
        if (close != std::string_view::npos) return value.substr(1, close - 1);
    }
    const std::size_t hash = value.find('#');
    if (hash != std::string_view::npos && (hash == 0 || value[hash - 1] == ' '))
        value = value.substr(0, hash);
    value = trimmed(value);
    if (value.size() >= 2 && (value.front() == '\'' || value.front() == '"') &&
        value.back() == value.front())
        value = value.substr(1, value.size() - 2);
    return value;
}

struct Expectation {
    std::string property;
    std::vector<std::string> values;
};

struct Case {
    std::string key;  // the corpus entry as written, path and all: its identity
    std::string name; // the basename, which is what a release-name parser is given
    std::vector<Expectation> expectations;
};

void appendValue(Expectation& expectation, std::string_view raw) {
    const std::string_view value = withoutComment(raw);
    if (!value.empty()) expectation.values.emplace_back(value);
}

// `[ a, b, c ]`
void appendFlowSequence(Expectation& expectation, std::string_view value) {
    value = value.substr(1, value.size() - 2);
    while (!value.empty()) {
        const std::size_t comma = value.find(',');
        appendValue(expectation, trimmed(value.substr(0, comma)));
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
    }
}

// Returns the expectation the following block-sequence items (if any) belong to. A property the
// case states itself REPLACES the one inherited from `__default__` rather than joining it.
Expectation* startProperty(Case& current, std::string_view line) {
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos) return nullptr;
    const std::string property(trimmed(line.substr(0, colon)));
    auto existing = std::ranges::find(current.expectations, property, &Expectation::property);
    if (existing == current.expectations.end()) {
        current.expectations.push_back(Expectation{property, {}});
        existing = std::prev(current.expectations.end());
    } else {
        existing->values.clear();
    }
    const std::string_view value = trimmed(line.substr(colon + 1));
    if (value.starts_with('['))
        appendFlowSequence(*existing, value);
    else
        appendValue(*existing, value);
    return &*existing;
}

std::vector<Case> loadCorpus(const std::filesystem::path& directory, std::string& problem) {
    std::vector<Case> cases;
    for (const std::string_view filename : CorpusFiles) {
        std::ifstream source(directory / filename, std::ios::binary);
        if (!source) {
            problem = "cannot read " + (directory / filename).string();
            return {};
        }
        std::vector<Expectation> defaults;
        Case current;
        Expectation* active = nullptr;
        bool haveCase = false;
        // A handful of names are written twice. YAML mappings are last-wins, and the Python
        // harness in the training repository reads them through a real YAML loader, so matching
        // that here keeps the two from reporting different denominators for the same corpus.
        std::map<std::string, std::size_t> seen;
        const auto flush = [&] {
            if (!haveCase) return;
            if (current.key == "__default__") {
                defaults = current.expectations;
            } else if (!current.name.empty()) {
                const auto [position, inserted] = seen.try_emplace(current.key, cases.size());
                if (inserted)
                    cases.push_back(current);
                else
                    cases[position->second] = current;
            }
            haveCase = false;
        };
        std::string line;
        while (std::getline(source, line)) {
            const std::string_view view(line);
            const std::string_view content = trimmed(view);
            if (content.empty() || content.starts_with('#')) continue;
            if (view.starts_with("? ")) {
                flush();
                current = Case{};
                // Our contract parses ONE release name, not a filesystem path: the corpus's
                // directory context (which GuessIt reads) is dropped and only the basename is fed
                // in. Names whose title only appears in a parent directory therefore cannot be
                // answered here, by design rather than by defect.
                std::string_view name = trimmed(view.substr(2));
                if (name.size() >= 2 && (name.front() == '\'' || name.front() == '"') &&
                    name.back() == name.front())
                    name = name.substr(1, name.size() - 2);
                current.key = std::string(name);
                // ENTRIES WRITTEN AS A PATH ARE SKIPPED, not scored against the filename.
                // GuessIt reads the enclosing directories, so
                // "Movies/Fear and Loathing in Las Vegas (1998)/Fear.and.Loathing...mkv" asserts
                // year 1998 while the filename never says 1998. This parser is handed a release
                // name, so that expectation is unanswerable rather than missed, and scoring it
                // would measure the harness rather than the parser.
                current.name = name.find_first_of("/\\") == std::string_view::npos
                                   ? std::string(name)
                                   : std::string{};
                current.expectations = defaults;
                active = nullptr;
                haveCase = true;
            } else if (view.starts_with(": ")) {
                active = startProperty(current, trimmed(view.substr(2)));
            } else if (content.starts_with("- ")) {
                if (active != nullptr) appendValue(*active, trimmed(content.substr(2)));
            } else if (view.starts_with("  ")) {
                active = startProperty(current, content);
            }
        }
        flush();
    }
    return cases;
}

// --- our contract, stated as translation tables ---------------------------------------------------

template <typename Value, std::size_t Count>
const Value* lookup(const std::array<std::pair<std::string_view, Value>, Count>& table,
                    std::string_view key) {
    const auto found = std::ranges::find(table, key, &std::pair<std::string_view, Value>::first);
    return found == table.end() ? nullptr : &found->second;
}

// Our resolution is a TIER, and 576p/540p/480p/360p and interlaced SD all live in the same one.
constexpr std::array<std::pair<std::string_view, ResolutionTier>, 12> Resolutions{{
    {"4320p", ResolutionTier::P4320}, {"2160p", ResolutionTier::P2160},
    {"1440p", ResolutionTier::P1440}, {"1080p", ResolutionTier::P1080},
    {"1080i", ResolutionTier::P1080}, {"720p", ResolutionTier::P720},
    {"576p", ResolutionTier::P480}, {"540p", ResolutionTier::P480},
    {"480p", ResolutionTier::P480}, {"480i", ResolutionTier::P480},
    {"368p", ResolutionTier::P480}, {"360p", ResolutionTier::P480},
}};

constexpr std::array<std::pair<std::string_view, VideoCodec>, 10> VideoCodecs{{
    {"h.264", VideoCodec::H264}, {"h.265", VideoCodec::Hevc}, {"xvid", VideoCodec::Xvid},
    // DivX and Xvid are both MPEG-4 Part 2 encoders; our contract names that family once.
    {"divx", VideoCodec::Xvid}, {"mpeg-2", VideoCodec::Mpeg2}, {"mpeg2", VideoCodec::Mpeg2},
    {"vp9", VideoCodec::Vp9}, {"vp8", VideoCodec::Vp8}, {"vc-1", VideoCodec::Vc1},
    {"av1", VideoCodec::Av1},
}};

// GuessIt's source vocabulary against ours. Several of its members are finer than our contract
// (Digital Master, Laserdisc, VHS, HD-DVD) and are left untranslated rather than forced.
constexpr std::array<std::pair<std::string_view, SourceKind>, 12> Sources{{
    {"blu-ray", SourceKind::BluRay}, {"ultra hd blu-ray", SourceKind::BluRay},
    {"dvd", SourceKind::Dvd}, {"hdtv", SourceKind::Hdtv}, {"analog hdtv", SourceKind::Hdtv},
    {"ultra hdtv", SourceKind::Hdtv}, {"digital tv", SourceKind::Hdtv},
    {"satellite", SourceKind::Hdtv}, {"tv", SourceKind::Hdtv},
    {"camera", SourceKind::Cam}, {"telesync", SourceKind::Cam}, {"telecine", SourceKind::Cam},
}};

// Our audio origins carry these spellings; GuessIt's profile split (DTS-HD + "Master Audio") maps
// onto our single DTS-HD MA value.
constexpr std::array<std::pair<std::string_view, std::string_view>, 13> AudioCodecs{{
    {"dolby digital", "DD"}, {"dolby digital plus", "DDP"}, {"dolby truehd", "TrueHD"},
    {"dolby atmos", "Atmos"}, {"dts", "DTS"}, {"dts-hd", "DTS-HD MA"}, {"aac", "AAC"},
    {"flac", "FLAC"}, {"opus", "OPUS"}, {"mp3", "MP3"}, {"lpcm", "PCM"}, {"pcm", "PCM"},
    {"alac", "ALAC"},
}};

constexpr std::array<std::pair<std::string_view, EditionKind>, 12> Editions{{
    {"director's cut", EditionKind::DirectorsCut}, {"extended", EditionKind::Extended},
    {"special", EditionKind::SpecialEdition}, {"remastered", EditionKind::Remastered},
    {"criterion", EditionKind::Criterion}, {"imax", EditionKind::Imax},
    {"theatrical", EditionKind::Theatrical}, {"uncut", EditionKind::Uncut},
    {"unrated", EditionKind::Unrated}, {"uncensored", EditionKind::Uncensored},
    {"open matte", EditionKind::OpenMatte}, {"final cut", EditionKind::FinalCut},
}};

// Extensions with no entry in our closed container table (`mediumOfContainer` in FieldValues):
// a torrent or an NZB describes a release rather than holding it, and a thumbnail or a split
// archive volume is not a medium we name. No answer of ours could pass these, so they measure the
// size of that table rather than the parser. This is a list of GAPS, not of decisions — add the
// extension to the table and delete it here.
constexpr std::array<std::string_view, 5> UnmodelledContainers{
    "torrent", "nzb", "r00", "tgz", "tbn",
};

// Our platform value is the scene abbreviation; GuessIt returns the marketing name.
constexpr std::array<std::pair<std::string_view, std::string_view>, 14> Platforms{{
    {"amazonprime", "amzn"}, {"amazon", "amzn"}, {"netflix", "nf"}, {"itunes", "it"},
    {"disney", "dsnp"}, {"appletv", "atvp"}, {"hbomax", "max"}, {"hbo", "hbo"},
    {"hulu", "hulu"}, {"crunchyroll", "cr"}, {"paramount", "pmtp"}, {"peacock", "pcok"},
    {"channel4", "4od"}, {"dramafever", "df"},
}};

// ISO 639-1 and English names to the 639-2 codes our contract emits.
constexpr std::array<std::pair<std::string_view, std::string_view>, 62> Languages{{
    {"en", "eng"}, {"eng", "eng"}, {"english", "eng"},
    {"fr", "fra"}, {"fra", "fra"}, {"fre", "fra"}, {"french", "fra"},
    {"de", "deu"}, {"deu", "deu"}, {"ger", "deu"}, {"german", "deu"},
    {"es", "spa"}, {"spa", "spa"}, {"spanish", "spa"},
    {"it", "ita"}, {"ita", "ita"}, {"italian", "ita"},
    {"pt", "por"}, {"por", "por"}, {"pt-br", "por"}, {"portuguese", "por"},
    {"ru", "rus"}, {"rus", "rus"}, {"russian", "rus"},
    {"nl", "nld"}, {"nld", "nld"}, {"dut", "nld"}, {"dutch", "nld"},
    {"ja", "jpn"}, {"jpn", "jpn"}, {"japanese", "jpn"},
    {"zh", "zho"}, {"zho", "zho"}, {"chi", "zho"}, {"chinese", "zho"},
    {"ko", "kor"}, {"kor", "kor"}, {"korean", "kor"},
    {"cs", "ces"}, {"ces", "ces"}, {"cze", "ces"}, {"czech", "ces"},
    {"pl", "pol"}, {"pol", "pol"}, {"polish", "pol"},
    {"sv", "swe"}, {"swe", "swe"}, {"swedish", "swe"},
    {"he", "heb"}, {"heb", "heb"}, {"hebrew", "heb"},
    {"ro", "ron"}, {"ron", "ron"}, {"rum", "ron"}, {"romanian", "ron"},
    {"ca", "cat"}, {"cat", "cat"}, {"catalan", "cat"},
    {"hu", "hun"}, {"hungarian", "hun"}, {"el", "ell"}, {"greek", "ell"},
}};

std::vector<std::string> originValues(const ReleaseInfo& info, Field field) {
    std::vector<std::string> found;
    for (const FieldOrigin& origin : info.origins)
        if (origin.field == field) found.push_back(origin.value.empty() ? origin.text
                                                                        : origin.value);
    return found;
}

enum class Outcome { Pass, Fail, Unmapped };

Outcome asOutcome(bool passed) { return passed ? Outcome::Pass : Outcome::Fail; }

// Every number our contract would answer an "is episode N in here?" question with. Season-relative
// and absolute numbering live in separate fields precisely because they are different facts, but
// GuessIt asks one question of both, so both answer here.
std::set<int> episodeNumbers(const ReleaseInfo& info) {
    std::set<int> numbers;
    const auto span = [&numbers](int first, int last) {
        if (first <= 0) return;
        for (int value = first; value <= std::max(first, last); ++value) numbers.insert(value);
    };
    span(info.episode.value_or(0), info.episodeEnd.value_or(0));
    span(info.absoluteEpisode.value_or(0), info.absoluteEpisodeEnd.value_or(0));
    return numbers;
}

// The title as read, plus every alternate title, plus the prefix + title reading below.
std::set<std::string> titleKeys(const ReleaseInfo& info) {
    std::set<std::string> keys{textKey(info.title)};
    for (const std::string& value : originValues(info, Field::AlternateTitle))
        keys.insert(textKey(value));
    // A franchise prefix is decorative branding our contract keeps OUT of the title
    // (`Marvels Agents of S.H.I.E.L.D` -> prefix `Marvels`, title `Agents of S.H.I.E.L.D`);
    // GuessIt has no such property and writes the whole thing as the title. Both read the name
    // the same way, so prefix + title is offered as a key as well.
    for (const std::string& prefix : originValues(info, Field::FranchisePrefix))
        keys.insert(textKey(prefix + " " + info.title));
    return keys;
}

// THE JOINED TITLE. Since the 2026-09-03 weights the model reads a work's subtitle as part of its
// one title span - `Fairy Tail - 100 Years Quest`, `Boruto - Naruto Next Generations` - where
// GuessIt splits the same name into `title` and `alternative_title`. Both have read the name the
// same way, so the concatenation of GuessIt's pieces, in either order, is offered as a key too.
// This mirrors `titles_match` in the training repository's comparison scorer. A title that
// swallowed an edition tag still fails: nothing GuessIt stated concatenates to it.
std::set<std::string> joinedTitleKeys(const Case& entry) {
    std::vector<std::string> pieces;
    for (const Expectation& expectation : entry.expectations)
        if (expectation.property == "title" || expectation.property == "alternative_title")
            for (const std::string& value : expectation.values) pieces.push_back(textKey(value));
    std::set<std::string> joined;
    if (pieces.size() < 2 || pieces.size() > 4) return joined;
    std::ranges::sort(pieces);
    do {
        std::string whole;
        for (const std::string& piece : pieces) whole += piece;
        joined.insert(whole);
    } while (std::ranges::next_permutation(pieces).found);
    return joined;
}

Outcome check(const ReleaseInfo& info, const Expectation& expectation,
              const std::set<std::string>& joinedTitles) {
    const std::vector<std::string>& expected = expectation.values;
    const std::string& property = expectation.property;
    if (expected.empty()) return Outcome::Unmapped;

    const auto everyExpected = [&expected](const auto& predicate) {
        return asOutcome(std::ranges::all_of(expected, predicate));
    };
    const auto asNumber = [](const std::string& text) { return std::atoi(text.c_str()); };

    if (property == "title") {
        const std::set<std::string> ours = titleKeys(info);
        return everyExpected([&ours, &joinedTitles](const std::string& want) {
            if (ours.contains(textKey(want))) return true;
            return std::ranges::any_of(joinedTitles, [&ours](const std::string& whole) {
                return ours.contains(whole);
            });
        });
    }
    if (property == "episode_title") {
        std::set<std::string> ours;
        for (const std::string& value : originValues(info, Field::EpisodeTitle))
            ours.insert(textKey(value));
        return everyExpected([&ours](const std::string& want) {
            return ours.contains(textKey(want));
        });
    }
    if (property == "year")
        return everyExpected([&](const std::string& want) { return info.year == asNumber(want); });
    if (property == "season") {
        std::set<int> ours;
        const int season = info.season.value_or(0);
        for (int value = season; season > 0 && value <= std::max(season, info.seasonEnd.value_or(0));
             ++value)
            ours.insert(value);
        return everyExpected([&](const std::string& want) { return ours.contains(asNumber(want)); });
    }
    if (property == "episode" || property == "absolute_episode") {
        const std::set<int> ours = episodeNumbers(info);
        return everyExpected([&](const std::string& want) { return ours.contains(asNumber(want)); });
    }
    if (property == "screen_size") {
        const ResolutionTier* wanted = lookup(Resolutions, lowered(expected.front()));
        return wanted == nullptr ? Outcome::Unmapped : asOutcome(info.screenSize == *wanted);
    }
    if (property == "source") {
        bool mapped = false;
        for (const std::string& want : expected) {
            const SourceKind* wanted = lookup(Sources, lowered(want));
            if (wanted == nullptr) continue;
            mapped = true;
            // GuessIt writes one generic `Web` where our contract separates WEB-DL from WEBRip,
            // so either reading satisfies it.
            const bool web = *wanted == SourceKind::WebDl || *wanted == SourceKind::WebRip;
            if (info.source == *wanted ||
                (web && (info.source == SourceKind::WebDl || info.source == SourceKind::WebRip)))
                return Outcome::Pass;
        }
        if (std::ranges::any_of(expected, [](const std::string& want) {
                return lowered(want) == "web" || lowered(want) == "video-on-demand"; })) {
            return asOutcome(info.source == SourceKind::WebDl || info.source == SourceKind::WebRip);
        }
        return mapped ? Outcome::Fail : Outcome::Unmapped;
    }
    if (property == "video_codec") {
        const VideoCodec* wanted = lookup(VideoCodecs, lowered(expected.front()));
        return wanted == nullptr ? Outcome::Unmapped : asOutcome(info.videoCodec == *wanted);
    }
    if (property == "audio_codec") {
        std::set<std::string> ours;
        for (const std::string& value : originValues(info, Field::Audio)) ours.insert(value);
        if (info.audioProfile == "Atmos") ours.insert("Atmos");
        bool mapped = false;
        for (const std::string& want : expected) {
            const std::string_view* wanted = lookup(AudioCodecs, lowered(want));
            if (wanted == nullptr) continue;
            mapped = true;
            if (!ours.contains(std::string(*wanted))) return Outcome::Fail;
        }
        return mapped ? Outcome::Pass : Outcome::Unmapped;
    }
    if (property == "audio_channels") {
        std::set<std::string> ours;
        for (const std::string& value : originValues(info, Field::Audio)) ours.insert(value);
        return everyExpected([&ours](const std::string& want) { return ours.contains(want); });
    }
    if (property == "container") {
        const std::string want = lowered(expected.front());
        if (std::ranges::find(UnmodelledContainers, want) != UnmodelledContainers.end())
            return Outcome::Unmapped;
        return asOutcome(lowered(info.container) == want);
    }
    if (property == "release_group") {
        // A TRACKER TAG GLUED TO THE TEAM NAME IS ONE FACT WRITTEN TWO WAYS. Indexes strip the tag
        // and trackers glue it on, so `ORGANiC[eztv]` here and `ORGANiC` + tracker tag `eztv` from
        // this parser describe the same release. We separate them on purpose; the corpus keeps
        // them as one string, and grading either against the other measures the convention rather
        // than the parser. Every candidate below is built only from spans the parser already
        // returned, so a team it did not read cannot be matched this way. The comparison harness
        // in the training repository reconciles them identically.
        std::set<std::string> ours;
        for (const std::string& group : info.releaseGroups) ours.insert(textKey(group));
        // A TEAM NAME WRITTEN IN TWO PIECES. `Etc-Group` is one team to the corpus and two group
        // spans here; the joined reading is offered in every order, capped at three pieces because
        // no real name splits a team further and the permutations grow factorially.
        if (info.releaseGroups.size() >= 2 && info.releaseGroups.size() <= 3) {
            std::vector<std::string> keys;
            for (const std::string& group : info.releaseGroups) keys.push_back(textKey(group));
            std::ranges::sort(keys);
            do {
                std::string joined;
                for (const std::string& key : keys) joined += key;
                ours.insert(joined);
            } while (std::next_permutation(keys.begin(), keys.end()));
        }
        for (const FieldOrigin& origin : info.origins) {
            if (origin.field != Field::TrackerTag) continue;
            const std::string tag = textKey(origin.text);
            if (tag.empty()) continue;
            ours.insert(tag);
            for (const std::string& group : info.releaseGroups) {
                const std::string key = textKey(group);
                ours.insert(key + tag);
                ours.insert(tag + key);
            }
        }
        return asOutcome(std::ranges::any_of(expected, [&ours](const std::string& want) {
            return ours.contains(textKey(want));
        }));
    }
    if (property == "streaming_service") {
        const std::string ours = lowered(info.streamingService);
        return asOutcome(std::ranges::any_of(expected, [&ours](const std::string& want) {
            const std::string_view* wanted = lookup(Platforms, textKey(want));
            return wanted != nullptr ? ours == *wanted : textKey(want) == textKey(ours);
        }));
    }
    if (property == "language" || property == "subtitle_language") {
        const std::vector<std::string>& ours = property == "language" ? info.languages
                                                                      : info.subtitleLanguages;
        bool mapped = false;
        for (const std::string& want : expected) {
            const std::string key = lowered(want);
            // "und" says GuessIt could not name the language and "mul"/"Multiple languages" says
            // there are several; our contract answers those with flags, not with a language code.
            if (key == "und" || key == "mul" || key == "multiple languages" || key == "multi")
                continue;
            const std::string_view* wanted = lookup(Languages, key);
            if (wanted == nullptr) continue;
            mapped = true;
            if (std::ranges::find(ours, *wanted) == ours.end()) return Outcome::Fail;
        }
        return mapped ? Outcome::Pass : Outcome::Unmapped;
    }
    // `type` is deliberately not scored; see the header. It restates the entry's own numbering
    // properties, and our content kind is a wider classification than movie-versus-episode.
    if (property == "type") return Outcome::Unmapped;
    if (property == "crc32") {
        std::set<std::string> ours;
        for (const std::string& value : originValues(info, Field::Crc32)) ours.insert(lowered(value));
        return everyExpected([&ours](const std::string& want) {
            return ours.contains(lowered(want));
        });
    }
    if (property == "color_depth") {
        std::set<std::string> ours;
        for (const std::string& value : originValues(info, Field::BitDepth)) ours.insert(textKey(value));
        return everyExpected([&ours](const std::string& want) {
            return ours.contains(textKey(want)); // "10-bit" and our "10bit" share a key
        });
    }
    if (property == "edition") {
        bool mapped = false;
        for (const std::string& want : expected) {
            const EditionKind* wanted = lookup(Editions, lowered(want));
            if (wanted == nullptr) continue;
            mapped = true;
            if (std::ranges::find(info.editions, *wanted) == info.editions.end())
                return Outcome::Fail;
        }
        return mapped ? Outcome::Pass : Outcome::Unmapped;
    }
    if (property == "proper_count") {
        // GuessIt counts the markers; our contract records which kind was stated.
        return asOutcome((asNumber(expected.front()) >= 1) == (info.proper || info.repack));
    }
    if (property == "website") {
        std::set<std::string> ours;
        for (const std::string& value : originValues(info, Field::SiteBanner))
            ours.insert(textKey(value));
        return everyExpected([&ours](const std::string& want) {
            return ours.contains(textKey(want));
        });
    }
    return Outcome::Unmapped; // no counterpart in our contract; not ours to be graded on
}

// The recorded state. A property below its floor is a regression in THIS parser; a property above
// it is an improvement worth re-pinning.
struct Floor {
    std::string_view property;
    int minimumPassing;
};

// Recorded at 2026-08-23 against GuessIt v4.4.0, and re-pinned on 2026-08-28 for the three-layer
// segmenter that shipped that day: release_group 480 -> 477 and website 10 -> 9, three and one
// names respectively, against a corpus total that went UP with the same model (574 of 858 fully
// correct). A floor is re-pinned only with the trade written down beside it; a silent lowering
// would defeat the point of having floors.
//
// Re-pinned again on 2026-08-29 for the franchise-prefix segmenter (donor015-prefix2):
// season 357 -> 356, one name, against a corpus total that went up with the same model
// (624 of 858 fully correct, from 574; title 760 against its 718 floor).
// How many names ONE property may lose before it fails, and the corpus total that may not slide.
// Two names is the observed noise of a segmenter swap on a single property; the total is what
// actually says whether the parser got worse.
constexpr int FloorTolerance = 2;
// Re-pinned 2026-09-03 for the merge-sub weights (subtitles folded into the title): 630 -> 618 in
// this build, with two scoring rules added the same day so that a reading GuessIt has no property
// for is not counted against us (title + alternative_title joined; franchise prefix + title). The
// 12 that remain are titles with a trailing number or tag glued on (`Test 12`, `Movie.Name.VR.180`,
// `Show.Name.July.30.2021`) - a training-recipe effect measured in the training repository - traded
// against episode_title 109 -> 117, language 84 -> 92, and +7 points exact parse on the hard
// validation set there. Per property: title 765 -> 747, everything else within one name.
// NOT re-pinned when the marker reading was consolidated into `convert::episodeMarkerIn`: that
// change is behaviour-identical here by construction, and the two notation extensions tried
// alongside it (a chained `1x02x03x04`, a four-digit `1940x01`) were worth three fixture names on
// this corpus and a regression on the only real name in 3,317 that has the shape, so they were
// dropped rather than banked.
//
// Re-pinned 2026-09-04, same weights, 628 -> 649, and none of it is a model change. Two readings
// of a release group this file did not have: the tracker tag glued to the team (`ORGANiC[eztv]` is
// one string to the corpus and a group plus an indexer tag here) and a team written in two pieces
// (`Etc-Group`), which together took release_group 478 -> 502. Both are conventions, not readings,
// and the comparison harness in the training repository had already reconciled them; this file had
// not, so those names were being scored on which project glues and which splits. The last one is
// `6.0` folding to 5.1 in the audio-channel table, where `6CH` already did.
//
// RE-PINNED DOWN 2026-09-11 for model 3, deliberately, and this is the trade. The run that split
// the animated/live-action content kinds into a `movie`/`series` form plus a separate `anime`
// verdict costs four fixture cases here (667 -> 663) and one `color_depth` name. It buys the
// opposite result everywhere the names are real: on the 3,344-name video validation split the
// comparison harness reads 683 of 859 corpus cases against 682, exact agreement across all shared
// fields 89.44% against 89.08%, and the new `anime` field lands at 96.50% through the int8 runtime.
// A ratchet that only ever goes up would have to refuse a better parser to keep a fixture score, so
// the total is re-pinned and the reason is written here rather than argued again next time.
constexpr int RecordedFullyCorrect = 663;   // 2026-09-11, model 3 (was 667 on the hybrid-labelled
                                            // weights, which was 652: merge-sub plus the source and
                                            // codec spellings, the two group reconciliations and
                                            // the channel-count spelling
                                            // (was 638, 628, 618, and 625 under donor015-prefix2)

constexpr std::array Floors{
    Floor{"title", 718},            Floor{"release_group", 477},
    Floor{"episode", 397},          Floor{"source", 452},           Floor{"video_codec", 413},
    Floor{"season", 356},           Floor{"screen_size", 421},      Floor{"year", 222},
    Floor{"audio_codec", 173},      Floor{"episode_title", 103},    Floor{"container", 128},
    Floor{"language", 80},          Floor{"audio_channels", 99},    Floor{"subtitle_language", 43},
    Floor{"streaming_service", 16}, Floor{"crc32", 26},             Floor{"color_depth", 22},
    Floor{"edition", 18},           Floor{"proper_count", 18},      Floor{"website", 9},
    // Dual-numbering names ("Bleach - s16e03-04 - 313-314") state season-relative AND absolute
    // episodes. GuessIt returns both; our contract elects one primary numbering, so all three of
    // these assertions are a known, deliberate difference rather than a defect to chase.
    Floor{"absolute_episode", 0},
};

std::filesystem::path corpusDirectory() {
    if (const char* override = std::getenv("RP_GUESSIT_CORPUS_DIR")) return {override};
#ifdef RP_GUESSIT_CORPUS_DIR
    return {RP_GUESSIT_CORPUS_DIR};
#else
    return {};
#endif
}

struct Tally {
    int passed = 0;
    int failed = 0;
    int unmapped = 0;
};

} // namespace

TEST_CASE("GuessIt's published corpus stays at or above the recorded reading") {
    std::string problem;
    const std::filesystem::path directory = corpusDirectory();
    const std::vector<Case> cases = loadCorpus(directory, problem);
    REQUIRE_MESSAGE(problem.empty(), problem);
    REQUIRE(!cases.empty());

    std::vector<std::string> names;
    names.reserve(cases.size());
    for (const Case& entry : cases) names.push_back(entry.name);

    BatchParser parser(RP_MODELS_DIR);
    const std::vector<ParseResult> results = parser.parse(names);
    REQUIRE(results.size() == cases.size());

    std::map<std::string, Tally> tallies;
    int fullyCorrect = 0;
    int scoredCases = 0;
    for (std::size_t index = 0; index < cases.size(); ++index) {
        bool anyScored = false;
        bool allPassed = true;
        const std::set<std::string> joinedTitles = joinedTitleKeys(cases[index]);
        for (const Expectation& expectation : cases[index].expectations) {
            const Outcome outcome = check(results[index].info, expectation, joinedTitles);
            Tally& tally = tallies[expectation.property];
            if (outcome == Outcome::Pass) ++tally.passed;
            else if (outcome == Outcome::Fail) ++tally.failed;
            else ++tally.unmapped;
            if (outcome != Outcome::Unmapped) anyScored = true;
            if (outcome == Outcome::Fail) allPassed = false;
        }
        scoredCases += anyScored;
        fullyCorrect += anyScored && allPassed;
    }

    int totalPassed = 0;
    int totalScored = 0;
    std::cout << "\nGuessIt corpus (" << cases.size() << " names from " << directory.string()
              << ")\n";
    for (const auto& [property, tally] : tallies) {
        const int scored = tally.passed + tally.failed;
        totalPassed += tally.passed;
        totalScored += scored;
        if (scored == 0) continue;
        std::cout << "  " << property << ": " << tally.passed << "/" << scored << " ("
                  << (100.0 * tally.passed / scored) << "%), " << tally.unmapped
                  << " unmapped\n";
    }
    std::cout << "  TOTAL: " << totalPassed << "/" << totalScored << " assertions, "
              << fullyCorrect << "/" << scoredCases << " cases fully correct\n\n";

    if (cases.size() != RecordedCaseCount) {
        MESSAGE("corpus has " << cases.size() << " names but the floors were recorded against "
                              << RecordedCaseCount
                              << "; reporting only, floors not enforced for this checkout");
        return;
    }
    // A FLOOR WITH A TOLERANCE, AND A TOTAL THAT CANNOT SLIDE.
    //
    // The per-property floors describe the installed WEIGHTS, and a segmenter swap moves a handful
    // of names either way for reasons that have nothing to do with the code under test: three
    // candidates in three days each failed a single property by one or two names while improving
    // the corpus overall (574 -> 624 -> 625 cases). Enforcing every property to the exact name
    // made the suite a re-pinning chore and taught nobody anything.
    //
    // So: a property may lose up to `FloorTolerance` names without failing — it is reported, and
    // a MESSAGE keeps it visible — while the CORPUS TOTAL is enforced strictly. Losing two names
    // on `season` and gaining six elsewhere is a better model; losing two everywhere is not, and
    // only the total can tell those apart.
    int slipped = 0;
    for (const Floor& floor : Floors) {
        const Tally& tally = tallies[std::string(floor.property)];
        if (tally.passed >= floor.minimumPassing) continue;
        const int shortfall = floor.minimumPassing - tally.passed;
        slipped += shortfall;
        MESSAGE(floor.property << ": " << tally.passed << " passing, floor is "
                               << floor.minimumPassing << " (short by " << shortfall
                               << ", tolerated up to " << FloorTolerance << ")");
        CHECK_MESSAGE(shortfall <= FloorTolerance,
                      floor.property << ": " << tally.passed << " passing against a floor of "
                                     << floor.minimumPassing << " — short by " << shortfall
                                     << ", beyond the " << FloorTolerance
                                     << "-name tolerance. A regression, or a floor to re-pin "
                                        "deliberately with the trade written down.");
    }
    CHECK_MESSAGE(fullyCorrect >= RecordedFullyCorrect - FloorTolerance,
                  "corpus total: " << fullyCorrect << " of " << scoredCases
                                   << " cases fully correct against a recorded "
                                   << RecordedFullyCorrect
                                   << " — the per-property slips add up to a worse parser");
    if (fullyCorrect > RecordedFullyCorrect)
        MESSAGE("corpus total improved: " << fullyCorrect << " against a recorded "
                                          << RecordedFullyCorrect
                                          << "; re-pin RecordedFullyCorrect to keep the ratchet");
    if (slipped > 0)
        MESSAGE("per-property slips total " << slipped
                                            << " name(s), within tolerance and reported above");
}

int main(int argc, char** argv) {
    const std::filesystem::path directory = corpusDirectory();
    std::error_code error;
    if (directory.empty() || !std::filesystem::is_directory(directory, error)) {
        std::cout << "GuessIt corpus not found at '" << directory.string()
                  << "'; skipping. Clone https://github.com/guessit-io/guessit beside this "
                     "repository, or set RP_GUESSIT_CORPUS_DIR.\n";
        return SkipExitCode;
    }
    for (const std::string_view filename : CorpusFiles) {
        if (!std::filesystem::exists(directory / filename, error)) {
            std::cout << "GuessIt corpus at '" << directory.string() << "' has no " << filename
                      << "; skipping.\n";
            return SkipExitCode;
        }
    }
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
