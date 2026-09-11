#pragma once

#include "neurelease/analysis.hpp"

#include <memory>
#include <string_view>

namespace neurelease::model {

// One instance owns reusable inference scratch and is therefore used by one thread at a time.
// Construct once per worker; model loading and kernel self-checks are intentionally not per parse.
class NameSegmenter {
  public:
    explicit NameSegmenter(std::string_view modelDirectory);
    NameSegmenter(std::string_view modelDirectory, const NameSegmenter& modelSource);
    ~NameSegmenter();
    NameSegmenter(NameSegmenter&&) noexcept;
    NameSegmenter& operator=(NameSegmenter&&) noexcept;
    NameSegmenter(const NameSegmenter&) = delete;
    NameSegmenter& operator=(const NameSegmenter&) = delete;

    [[nodiscard]] Analysis analyze(std::string_view name);

    // Optional process-level persistence for a kernel that fails its startup self-check. Set before
    // constructing a segmenter. The public facade owns the user-facing policy.
    static void rememberKernelFailures(std::string_view file);

  private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

} // namespace neurelease::model
