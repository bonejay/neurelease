// Two-level release-name segmenter, forward pass only.
//
// Loads the artifact written by training/export_segmenter_weights.py and runs it. No oneDNN, no
// ONNX Runtime, no BLAS - only the standard library and the SIMD kernels in this directory,
// which is the bar docs/CPU_MODEL_DEPLOYMENT.md sets for any learned model entering the parser.
//
// The network shape is read from the artifact, not compiled in: a retrained model of a different
// width or depth loads without a rebuild, and an artifact this code cannot honour (an unknown
// token architecture, a missing tensor) fails loudly at construction with the missing piece
// named. What IS fixed here is the family of architectures the training code can produce — a
// character convolution stack, max-pooled into pseudo-tokens, then a transformer or convolution
// token stage, with span heads and per-name classification heads.
//
// Precision: trunk matmuls run int8 (weights per-output-channel symmetric as exported,
// activations quantised per tensor on the fly, exact int32 accumulation), everything else —
// normalisation, GELU, softmax, attention scores, residuals, and the decision heads — runs
// float32. Measured in training, int8 weights are exactly lossless on the validation split; the
// heads stay float because a binary decision flips outright on a small logit shift.
// Float32Reference runs the whole model in float32 from dequantised weights: slower, kept forever
// as the oracle the int8 path and every new kernel are verified against, name by name.

#pragma once

#include "model/encoder.hpp"
#include "model/weights.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace neurelease::model::floats {
struct Table;
}

namespace neurelease::model {

enum class Precision : std::uint8_t { Int8, Float32Reference };

// How one prediction was produced. Nobody needs this to USE the result — it exists so a caller
// that logs, profiles, or debugs can do it from the return value alone: which kernel actually
// ran (and whether the startup self-check demoted it to the safe scalar fallback), what the
// input looked like after encoding, and where the time went. Costs four clock reads per name.
struct RunDiagnostics {
    const char *kernelPath = "";  // "scalar", "avx2", "avx2vnni", "avx512vnni"
    bool kernelDemoted = false;   // a SIMD path failed its self-check; running on scalar
    float characterMicros = 0.0F; // embeddings + character convolutions
    float tokenMicros = 0.0F;     // pooling + transformer
    float totalMicros = 0.0F;     // whole forward pass; heads are total minus the two stages
};

// What the model says about one name. Probabilities, not logits: every consumer wants a decision
// or a confidence, and softmax at the end costs nothing at this size.
struct Prediction {
    // Per run: probability that a new segment starts here. Index 0 is what the model says, but
    // decoding always starts a segment at run 0 regardless.
    std::vector<float> splitStart;
    // Per run: distribution over span types; index 0 is "nothing here", the rest follow
    // Segmenter::spanTypes().
    std::vector<std::vector<float>> typeScores;
    // Per whole-name field, in Segmenter::globalFields() order: distribution over its classes.
    std::vector<std::vector<float>> globals;
    RunDiagnostics info;
};

// One decoded span over source codepoints. `type` indexes Segmenter::spanTypes();
// `confidence` is the winning type's mean probability over the segment's runs.
struct PredictedSpan {
    int type = 0;
    std::int32_t begin = 0;
    std::int32_t end = 0;
    float confidence = 0.0F;
};

class Segmenter {
    public:
    // Prepares the forward pass: dequantises embeddings, repacks the quantised matrices for the
    // kernels, and resolves the network shape. Throws std::runtime_error naming what is wrong
    // when the artifact does not describe a network this code can run.
    explicit Segmenter(const Weights &weights, Precision precision = Precision::Int8);

    // Thread-safe; scratch buffers are per thread.
    [[nodiscard]] Prediction run(const EncodedName &name) const;

    // Merge runs into typed spans the way training was evaluated: a segment starts where the
    // split head says so, its type is the argmax of the mean type distribution over its runs,
    // and "nothing here" spans are dropped.
    [[nodiscard]] static std::vector<PredictedSpan> spans(const EncodedName &name,
                                                          const Prediction &prediction);

    [[nodiscard]] const std::vector<std::string> &spanTypes() const noexcept;
    struct GlobalField {
        std::string name;
        std::vector<std::string> values;
    };
    [[nodiscard]] const std::vector<GlobalField> &globalFields() const noexcept;
    [[nodiscard]] Precision precision() const noexcept;

    // Intermediate stages, exposed so the reference tests can bisect a mismatch to the stage that
    // introduced it rather than only seeing that the final logits differ. Row-major buffers.
    void characterStage(const EncodedName &name, std::vector<float> &hidden) const;
    void pool(const EncodedName &name, const std::vector<float> &hidden,
              std::vector<float> &pooled) const;
    void tokenStage(const EncodedName &name, const std::vector<float> &pooled,
                    std::vector<float> &tokenHidden, std::vector<float> &summary) const;

    [[nodiscard]] int charChannels() const noexcept;
    [[nodiscard]] int tokenChannels() const noexcept;

    private:
    struct Prepared;
    std::shared_ptr<const Prepared> prepared_;
};

// Where the milliseconds go, accumulated per thread across every run() since the last reset.
// Costs a few timestamps per name; exists so a performance regression is attributable to a stage
// from the benchmark output alone, without a profiler session.
struct ProfileCounters {
    double convolutionSeconds = 0;   // character convolutions: quantise + matvec + norm + GELU
    double matmulSeconds = 0;        // every applyMatrix: projections, attention QKV, FFN, heads
    double attentionSeconds = 0;     // scores, softmax and context — the O(S^2) float core
    // How many head calls attentionSeconds covers, so per-call cost is a DIVISION -
    // attentionSeconds / attentionCalls - rather than an inference from a guessed head count.
    long attentionCalls = 0;
    double totalSeconds = 0;
    long names = 0;
};
[[nodiscard]] ProfileCounters profileCounters() noexcept;

// THE FLOAT SIDE OF THE FORWARD PASS, in its faster form.
//
// Layer norm, GELU, softmax and the two attention primitives have an AVX2 + FMA implementation that
// is eight lanes wide where the baseline is four or none. It is off by default: its GELU is the tanh
// approximation rather than the erf the model trained with, so results move in the last digits, and
// that is a change to ask for rather than to inherit. Returns whether it is now actually in use -
// false means this CPU (or this build) has no AVX2 and the baseline is still running.
//
// DEFAULT: Baseline. Not because the AVX2 table is slower - it is 1.20x faster - but because no
// eight-wide float path can be bit-identical to a four-wide one (different addition order, and FMA
// rounds once where multiply-then-add rounds twice), so it moves 0.3% of parses. A speed-up that
// changes answers is a flag, not a default. NEURELEASE_SEGMENTER_FLOATS=avx2 or =tanhgelu turns it
// on; an explicit call here always beats the environment.
//
// Process-wide, and not safe to flip while a parse is in flight.
//
//   Baseline      - scalar and SSE2, the reference.
//   Avx2          - eight lanes and FMA, with an erf GELU accurate to 1e-7. Same answers.
//   Avx2FastGelu  - as above but the tanh GELU: quicker again, and it changes about 1% of parses.
enum class FloatMode { Baseline, Avx2, Avx2FastGelu };
bool useFastFloats(FloatMode mode) noexcept;
[[nodiscard]] bool fastFloatsInUse() noexcept;

// The reference float table - scalar and SSE2 - so a test can hold the AVX2 one against it directly
// rather than inferring correctness from how many names come out differently.
[[nodiscard]] const floats::Table *baselineFloats() noexcept;

// RUN THE DECISION HEADS EXACTLY while the trunk runs the fast u8 kernel. Default OFF - measured
// useless. The hypothesis was that the heads, feeding argmaxes directly, would turn small errors into
// changed answers; the measurement says the divergence count is IDENTICAL with and without at every
// zero point, because the fast kernel's errors enter in the trunk's representations and exact heads
// just map the shifted inputs to the same different answers. Kept so the measurement stays
// repeatable.
//
// Process-wide, and not safe to flip while a parse is in flight.
void useExactHeads(bool wanted) noexcept;
[[nodiscard]] bool exactHeadsInUse() noexcept;
void resetProfileCounters() noexcept;

} // namespace neurelease::model
