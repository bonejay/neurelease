// Title assembly: joining the model's title spans back into readable text. Separators inside a
// title become spaces, a subtitle span the model still emits is folded into the title, and nothing
// is respelled -- a typo in the name stays a typo in the title, because canonical spelling is a
// catalogue lookup that belongs to the caller.

#include "convert/field_values.hpp"
#include "convert/generated/aliases.hpp"

#include "text/regex.hpp"
#include "text/unicode.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace neurelease::convert {
namespace {

std::vector<std::string> words(std::string_view value) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin < value.size()) {
        while (begin < value.size() && value[begin] == ' ') ++begin;
        if (begin == value.size()) break;
        const std::size_t end = value.find(' ', begin);
        result.emplace_back(value.substr(begin, end == std::string_view::npos ? value.size() - begin
                                                                             : end - begin));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return result;
}

std::string join(const std::vector<std::string>& values) {
    std::string result;
    for (const std::string& value : values) {
        if (!result.empty()) result.push_back(' ');
        result += value;
    }
    return result;
}

bool hasLatinWord(std::string_view value) {
    int run = 0;
    for (std::size_t at = 0; at < value.size();) {
        const char32_t cp = text::decodeAt(value, at);
        if (text::isLatinLetter(cp)) {
            if (++run >= 2) return true;
        } else {
            run = 0;
        }
    }
    return false;
}

bool hasForeignScript(std::string_view value) {
    for (std::size_t at = 0; at < value.size();) {
        if (text::isForeignScript(text::decodeAt(value, at))) return true;
    }
    return false;
}

bool hasLetterOrNumber(std::string_view value) {
    for (std::size_t at = 0; at < value.size();) {
        const char32_t cp = text::decodeAt(value, at);
        if (text::isLetter(cp) || text::isDecimalDigit(cp)) return true;
    }
    return false;
}

std::string stripEdgePunctuation(std::string_view value) {
    static const text::Regex edges(
        R"(^[\s/|:;,\.\-\x{2013}\x{2014}]+|[\s/|:;,\.\-\x{2013}\x{2014}]+$)");
    std::string result(value);
    for (;;) {
        const text::Match match = edges.match(result);
        if (!match) break;
        result.erase(match.capturedStart(), match.capturedEnd() - match.capturedStart());
    }
    return result;
}

} // namespace

std::string preferLatinTitle(std::string_view title) {
    const std::string simplified = text::simplified(title);
    const std::vector<std::string> tokens = words(simplified);
    bool foreign = false;
    bool latinWord = false;
    for (const std::string& token : tokens) {
        foreign = foreign || hasForeignScript(token);
        latinWord = latinWord || hasLatinWord(token);
    }
    if (!foreign || !latinWord) return simplified;

    std::vector<std::string> kept;
    for (const std::string& token : tokens) {
        std::string stripped;
        for (std::size_t at = 0; at < token.size();) {
            const char32_t cp = text::decodeAt(token, at);
            if (text::isForeignScript(cp)) stripped.push_back(' ');
            else text::appendUtf8(stripped, cp);
        }
        std::string candidate = stripEdgePunctuation(text::simplified(stripped));
        if (!candidate.empty() && hasLetterOrNumber(candidate)) kept.push_back(std::move(candidate));
    }
    return kept.empty() ? simplified : join(kept);
}

std::string titleText(std::string_view raw) {
    std::string value(raw);
    std::ranges::replace(value, '.', ' ');
    std::ranges::replace(value, '_', ' ');
    value = text::simplified(value);
    std::vector<std::string> parts = words(value);
    // A TITLE WRITTEN TWICE collapses to one copy: `Some.Movie.Some.Movie` -> `Some Movie`. The
    // halves must be at least TWO words each, because a one-word half is far more often a real
    // doubled title than a duplication - `Tari Tari`, `Duran Duran`, `Sing Sing`, `Boum Boum`.
    // Collapsing those silently truncated the title while the span itself was correct, which is
    // how it survived: the model segmented `Tari Tari` and the value came out `Tari`.
    // Known limitation: a two-word title doubled on purpose (`New York New York`) still collapses.
    // Telling that apart from a duplicated `Some Movie` needs knowledge of the work, which this
    // layer does not have.
    if (parts.size() >= 4 && parts.size() % 2 == 0) {
        const auto middle = parts.begin() + static_cast<std::ptrdiff_t>(parts.size() / 2);
        if (std::equal(parts.begin(), middle, middle, parts.end())) parts.erase(middle, parts.end());
    }
    value = join(parts);
    static const text::Regex trailing(R"([\s(\[{:;|/\-]+$)");
    if (const text::Match match = trailing.match(value); match)
        value.erase(match.capturedStart(), match.capturedEnd() - match.capturedStart());
    return preferLatinTitle(text::simplified(value));
}

std::string groupText(std::string_view raw) {
    std::string candidate(text::trimmed(raw));
    static const text::Regex enclosing(R"(^[\[({]\s*|\s*[\])}]$)");
    for (;;) {
        const text::Match match = enclosing.match(candidate);
        if (!match) break;
        candidate.erase(match.capturedStart(), match.capturedEnd() - match.capturedStart());
    }
    candidate = std::string(text::trimmed(candidate));
    while (candidate.starts_with('-')) candidate.erase(candidate.begin());
    static const text::Regex container(R"(\.(mkv|mp4|avi|ts|m2ts|wmv|mov|mp3|flac)$)", true);
    if (const text::Match match = container.match(candidate); match)
        candidate.erase(match.capturedStart(), match.capturedEnd() - match.capturedStart());
    static const text::Regex digitsOnly(R"(^\d+$)");
    return candidate.empty() || digitsOnly.match(candidate) ? std::string{} : candidate;
}

bool isNeverAGroup(std::string_view word) {
    const std::string key = text::asciiLower(word);
    if (std::ranges::binary_search(generated::rejectedGroups, key)) return true;
    static const text::Regex coverage(R"(^(?:S\d{1,2}(?:E\d{1,4})?|E\d{1,4}|EP\d{1,3})$)", true);
    return static_cast<bool>(coverage.match(word));
}

} // namespace neurelease::convert
