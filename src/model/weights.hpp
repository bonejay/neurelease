// Loader for the RPSEG2 weight artifact written by training/export_segmenter_weights.py.
//
// A generic tensor container on purpose: the file format knows nothing about the network, so a
// retrained model of a different width, depth, or token architecture loads without touching this
// code. Everything architecture-specific — which tensors must exist, what their shapes mean — is
// asked by SegmenterModel when it prepares the forward pass, where a missing piece can be named
// in the error.
//
// No JSON, no compression, no third-party reader. Fixed-width little-endian records, documented
// in the exporter's docstring, parsed in the ~100 lines of SegmenterWeights.cpp.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace neurelease::model {

// One tensor as stored. Exactly one of the three payloads is populated, matching the dtype in the
// file: float32 weights, int8 codes with one float scale per output channel (row), or UTF-8 text
// (the exporter uses text tensors for label vocabularies).
struct Tensor {
    std::vector<std::int64_t> shape;
    std::vector<float> floats;
    std::vector<std::int8_t> codes;
    std::vector<float> scales; // one per shape[0] when quantised
    std::string text;

    [[nodiscard]] bool quantised() const noexcept { return !codes.empty(); }
    [[nodiscard]] std::int64_t rows() const noexcept { return shape.empty() ? 0 : shape[0]; }
    [[nodiscard]] std::int64_t count() const noexcept;
    // Width of one row: the product of every dimension after the first. For the artifact's
    // per-output-channel quantisation this is the extent one scale covers.
    [[nodiscard]] std::int64_t rowWidth() const noexcept;
};

class Weights {
    public:
    // Reads and validates the whole artifact. Throws std::runtime_error naming the offending
    // field when the magic, version, a bound, or a payload size does not hold together — a
    // truncated or foreign file must fail loudly, never misread.
    static Weights load(const std::string &path);

    [[nodiscard]] bool has(std::string_view name) const noexcept;
    // Throws std::runtime_error naming the tensor when it is absent, so a caller never has to
    // produce its own "which tensor was missing" diagnostics.
    [[nodiscard]] const Tensor &tensor(std::string_view name) const;
    // Convenience accessors for the meta tensors; same missing-tensor behaviour.
    [[nodiscard]] const std::string &text(std::string_view name) const;
    [[nodiscard]] float scalar(std::string_view name) const;

    [[nodiscard]] const std::vector<std::string> &names() const noexcept;

    private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;
};

} // namespace neurelease::model
