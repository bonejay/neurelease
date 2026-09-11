#include "text/regex.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <stdexcept>

namespace neurelease::text {

std::size_t Match::capturedStart(int group) const noexcept {
    if (!matched_ || group < 0 || group >= groupCount_) return npos;
    return offsets_[static_cast<std::size_t>(group) * 2];
}

std::size_t Match::capturedEnd(int group) const noexcept {
    if (!matched_ || group < 0 || group >= groupCount_) return npos;
    return offsets_[static_cast<std::size_t>(group) * 2 + 1];
}

std::string_view Match::captured(int group) const noexcept {
    const std::size_t begin = capturedStart(group);
    const std::size_t end = capturedEnd(group);
    if (begin == npos || end == npos || begin > end) return {};
    return subject_.substr(begin, end - begin);
}

struct Regex::Impl {
    pcre2_code* code = nullptr;
    // One match-data block per compiled pattern would be shared mutable state across threads;
    // per-call allocation would be a malloc on the hottest path. Thread-local, keyed by nothing —
    // a match-data block only depends on the ovector SIZE, which is fixed at MaxGroups for the
    // whole library — so one block per thread serves every pattern.
    static pcre2_match_data* threadMatchData() {
        // Deliberately never destroyed: freeing thread_local storage at thread exit corrupts the
        // heap on the MinGW toolchain this library is developed against (documented at length in
        // the training repository), and one block per thread is bytes, not megabytes.
        thread_local pcre2_match_data* data =
            pcre2_match_data_create(Match::MaxGroups, nullptr);
        return data;
    }

    ~Impl() {
        if (code != nullptr) pcre2_code_free(code);
    }
};

Regex::Regex(std::string_view pattern, Options options) : impl_(std::make_unique<Impl>()) {
    int error = 0;
    PCRE2_SIZE errorOffset = 0;
    // UTF for multi-byte sequences in patterns and subjects; UCP so \b, \w and \d classify by
    // Unicode property rather than by ASCII alone — a word boundary beside an accented letter is
    // not a boundary.
    std::uint32_t flags = PCRE2_UTF | PCRE2_UCP;
    if (options.caseInsensitive) flags |= PCRE2_CASELESS;
    impl_->code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                                flags, &error, &errorOffset, nullptr);
    if (impl_->code == nullptr) {
        PCRE2_UCHAR message[256];
        pcre2_get_error_message(error, message, sizeof(message));
        throw std::invalid_argument("neurelease regex: " +
                                    std::string(reinterpret_cast<char*>(message)) + " at offset " +
                                    std::to_string(errorOffset) + " in: " + std::string(pattern));
    }
    // JIT is an optimisation, not a requirement: where it is unavailable the interpreter runs the
    // same pattern, so the return value is deliberately ignored.
    pcre2_jit_compile(impl_->code, PCRE2_JIT_COMPLETE);
}

Regex::Regex(std::string_view pattern, bool caseInsensitive)
    : Regex(pattern, Options{caseInsensitive}) {}

Regex::~Regex() = default;
Regex::Regex(Regex&&) noexcept = default;
Regex& Regex::operator=(Regex&&) noexcept = default;

Match Regex::match(std::string_view subject, std::size_t from) const {
    Match result;
    result.subject_ = subject;
    if (from > subject.size()) return result;
    pcre2_match_data* data = Impl::threadMatchData();
    const int rc = pcre2_match(impl_->code, reinterpret_cast<PCRE2_SPTR>(subject.data()),
                               subject.size(), from, 0, data, nullptr);
    if (rc < 0) return result; // PCRE2_ERROR_NOMATCH and friends: simply no match
    result.matched_ = true;
    // rc == 0 means the ovector was too small for every group; MaxGroups of them still arrived.
    result.groupCount_ = rc == 0 ? Match::MaxGroups : rc;
    const PCRE2_SIZE* vector = pcre2_get_ovector_pointer(data);
    for (int group = 0; group < result.groupCount_ && group < Match::MaxGroups; ++group) {
        result.offsets_[group * 2] = vector[group * 2];
        result.offsets_[group * 2 + 1] = vector[group * 2 + 1];
    }
    return result;
}

} // namespace neurelease::text
