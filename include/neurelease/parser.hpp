#pragma once

#include "neurelease/analysis.hpp"
#include "neurelease/release_info.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace neurelease {

enum class Accuracy : std::uint8_t { Exact = 0, Fast = 1 };

// Accuracy selects process-wide kernel dispatch. Set during startup, before parsers are used; it is
// intentionally not a parser property because all instances share immutable packed model weights
// and the process-wide CPU dispatch table.
void setAccuracy(Accuracy accuracy) noexcept;
[[nodiscard]] Accuracy accuracy() noexcept;

struct ParseResult {
    ReleaseInfo info;
    Analysis analysis;
};

class Parser {
  public:
    explicit Parser(std::string_view modelDirectory);
    ~Parser();
    Parser(Parser&&) noexcept;
    Parser& operator=(Parser&&) noexcept;
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    // One Parser owns reusable scratch and is used by one thread at a time.
    [[nodiscard]] ParseResult parse(std::string_view name);

    static void rememberKernelFailures(std::string_view file);

  private:
    Parser(std::string_view modelDirectory, const Parser& modelSource);
    friend class BatchParser;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

class BatchParser {
  public:
    struct Timing {
        std::size_t names = 0;
        int threads = 0;
        int buckets = 0;
        double wallMilliseconds = 0.0;
        double encodeMilliseconds = 0.0;
        double modelMilliseconds = 0.0;
        double convolutionMilliseconds = 0.0;
        double matmulMilliseconds = 0.0;
        double attentionMilliseconds = 0.0;
        double bucketPaddingPercent = 0.0;

        [[nodiscard]] double averageWallMicrosPerName() const noexcept {
            return names == 0 ? 0.0 : wallMilliseconds * 1000.0 / static_cast<double>(names);
        }
    };

    explicit BatchParser(std::string_view modelDirectory, int threads = 0);
    ~BatchParser();
    BatchParser(BatchParser&&) noexcept;
    BatchParser& operator=(BatchParser&&) noexcept;
    BatchParser(const BatchParser&) = delete;
    BatchParser& operator=(const BatchParser&) = delete;

    [[nodiscard]] int threads() const noexcept;
    void setBucketSize(int names) noexcept;
    [[nodiscard]] int bucketSize() const noexcept;
    [[nodiscard]] Timing lastTiming() const noexcept;

    // Results preserve input order. The call owns all parallelism and is not itself re-entrant.
    [[nodiscard]] std::vector<ParseResult> parse(std::span<const std::string> names);

  private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace neurelease
