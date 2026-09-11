// A thin wall around PCRE2, exposing exactly the subset this library uses.
//
// The contract:
//   * UTF-8 everywhere. Subjects are std::string_view of UTF-8; offsets are BYTE offsets. Pattern
//     escapes address codepoints (\x{FF1A} is the fullwidth colon) whatever their encoded width.
//   * A Match BORROWS its subject. captured() views point into the caller's buffer and live only as
//     long as it does — the compiler cannot enforce that, so the convention is: consume a match
//     before the subject moves.
//   * Compile once, match often. Every pattern in this library is a function-local static; compiling
//     a pattern per call has been measured at a quarter of an entire engine's runtime, so the
//     constructor JIT-compiles eagerly and match() is allocation-free after the first call per
//     thread.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace neurelease::text {

class Regex;

// One match attempt's result. Group 0 is the whole match; groups that did not participate report
// begin == end == npos and empty views.
class Match {
  public:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    [[nodiscard]] bool hasMatch() const noexcept { return matched_; }
    explicit operator bool() const noexcept { return matched_; }

    // Byte offsets into the subject the match ran against.
    [[nodiscard]] std::size_t capturedStart(int group = 0) const noexcept;
    [[nodiscard]] std::size_t capturedEnd(int group = 0) const noexcept;
    [[nodiscard]] std::string_view captured(int group = 0) const noexcept;

  private:
    friend class Regex;
    bool matched_ = false;
    int groupCount_ = 0;
    std::string_view subject_;
    // begin/end pairs, group-indexed, straight from the PCRE2 ovector.
    static constexpr int MaxGroups = 16;
    std::size_t offsets_[MaxGroups * 2] = {};
};

// Namespace-scope on purpose: a nested option struct with default member initializers cannot be
// used as a default argument inside its own enclosing class.
struct RegexOptions {
    bool caseInsensitive = false;
};

class Regex {
  public:
    using Options = RegexOptions;

    // Compiles eagerly, JIT included. An invalid pattern is a programming error in this library —
    // every pattern is a literal — so it throws std::invalid_argument with PCRE2's message rather
    // than limping along as a never-matching object.
    explicit Regex(std::string_view pattern, RegexOptions options = {});
    Regex(std::string_view pattern, bool caseInsensitive);
    ~Regex();
    Regex(Regex&&) noexcept;
    Regex& operator=(Regex&&) noexcept;
    Regex(const Regex&) = delete;
    Regex& operator=(const Regex&) = delete;

    [[nodiscard]] Match match(std::string_view subject, std::size_t from = 0) const;

    // The globalMatch loop, flattened: the next match at or after `from`, with the caller advancing
    // `from` to capturedEnd() (plus one byte if the match was empty, or it would never move).
    [[nodiscard]] Match matchAfter(std::string_view subject, std::size_t from) const {
        return match(subject, from);
    }

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neurelease::text
