#include "model/segmenter_model.hpp"

#include "model/kernels/floats.hpp"

#include "model/kernels/kernels.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <atomic>
#include <string>
#include <mutex>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#define NEURELEASE_MODEL_SSE2 1
#include <emmintrin.h>
#endif

namespace neurelease::model {

namespace {

constexpr float NormEpsilon = 1e-5F; // torch.nn.LayerNorm default, shared by ChannelNorm

inline float gelu(float value) noexcept {
    // Exact erf GELU — torch.nn.GELU's default, and what both trunks and heads trained with.
    return 0.5F * value * (1.0F + std::erf(value * 0.70710678118654752F));
}

// GELU by table for the int8 path. std::erf is a ~47 ns library call on MinGW and the forward
// pass makes ~80,000 of them per name — measured, that was a third of the whole model. Piecewise
// linear over [-8, 8] at step 1/256: the worst-case error is h^2/8 * max|gelu''| < 1e-6, far
// below the int8 activation step itself. Outside the table GELU IS its asymptote in float32.
// The float32 reference path keeps the exact erf: it is the oracle, not the product.
struct GeluTable {
    std::array<float, 4098> values{};
    GeluTable() noexcept {
        for (std::size_t at = 0; at < values.size(); ++at) {
            values[at] = gelu(static_cast<float>(at) * (1.0F / 256.0F) - 8.0F);
        }
    }
};

inline float geluFast(float value) noexcept {
    static const GeluTable table;
    if (value <= -8.0F) {
        return 0.0F;
    }
    if (value >= 8.0F) {
        return value;
    }
    const float position = (value + 8.0F) * 256.0F;
    const int index = static_cast<int>(position);
    const float fraction = position - static_cast<float>(index);
    const float low = table.values[static_cast<std::size_t>(index)];
    const float high = table.values[static_cast<std::size_t>(index) + 1];
    return low + (high - low) * fraction;
}

// dot and y += a*x over floats, 4-wide with two accumulators on x86-64, scalar elsewhere.
// The compiler cannot do this itself: without -ffast-math it may not reassociate a float
// reduction, so the plain loop stays scalar — measured, the heads and attention cost real
// milliseconds that way.
float dotF32Baseline(const float *a, const float *b, int count) noexcept {
    float sum = 0.0F;
    int at = 0;
#ifdef NEURELEASE_MODEL_SSE2
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    for (; at + 8 <= count; at += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + at), _mm_loadu_ps(b + at)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + at + 4), _mm_loadu_ps(b + at + 4)));
    }
    __m128 acc = _mm_add_ps(acc0, acc1);
    acc = _mm_add_ps(acc, _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(1, 0, 3, 2)));
    acc = _mm_add_ps(acc, _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_cvtss_f32(acc);
#endif
    for (; at < count; ++at) {
        sum += a[at] * b[at];
    }
    return sum;
}

void addScaledBaseline(float *destination, const float *source, float weight,
                       int count) noexcept {
    int at = 0;
#ifdef NEURELEASE_MODEL_SSE2
    const __m128 factor = _mm_set1_ps(weight);
    for (; at + 4 <= count; at += 4) {
        _mm_storeu_ps(destination + at,
                      _mm_add_ps(_mm_loadu_ps(destination + at),
                                 _mm_mul_ps(_mm_loadu_ps(source + at), factor)));
    }
#endif
    for (; at < count; ++at) {
        destination[at] += source[at] * weight;
    }
}

// exp by range reduction and a degree-6 Taylor polynomial: exp(x) = 2^i * exp(f) with
// i = round(x / ln 2), |f| <= 0.3466, worst relative error ~1e-7. For the int8 path's softmax
// only — std::exp is another ~40 ns MinGW library call, at tens of thousands of calls per name.
// The float32 reference keeps std::exp.
inline float expFast(float value) noexcept {
    value = std::max(value, -87.0F);
#ifdef NEURELEASE_MODEL_SSE2
    const int exponent = _mm_cvt_ss2si(_mm_set_ss(value * 1.44269504F));
#else
    const int exponent = static_cast<int>(std::lrintf(value * 1.44269504F));
#endif
    const float f = value - static_cast<float>(exponent) * 0.6931471805599453F;
    const float p =
        1.0F + f * (1.0F + f * (0.5F + f * (0.16666667F +
                    f * (0.041666668F + f * (0.008333334F + f * 0.0013888889F)))));
    return p * std::bit_cast<float>(static_cast<std::uint32_t>(exponent + 127) << 23);
}

void softmaxInPlaceBaseline(float *values, int count, bool fast) noexcept {
    float largest = values[0];
    for (int at = 1; at < count; ++at) {
        largest = std::max(largest, values[at]);
    }
    float total = 0.0F;
    for (int at = 0; at < count; ++at) {
        values[at] = fast ? expFast(values[at] - largest) : std::exp(values[at] - largest);
        total += values[at];
    }
    const float inverse = 1.0F / total;
    for (int at = 0; at < count; ++at) {
        values[at] *= inverse;
    }
}


// LayerNorm over one row: y = (x - mean) / sqrt(var + eps) * weight + bias, variance biased,
// exactly torch. Reads `input`, writes `output`; the two may alias.
void layerNormBaseline(const float *input, int width, const float *weight, const float *bias,
                       float *output) noexcept {
    float mean = 0.0F;
    for (int at = 0; at < width; ++at) {
        mean += input[at];
    }
    mean /= static_cast<float>(width);
    float variance = 0.0F;
    for (int at = 0; at < width; ++at) {
        const float centred = input[at] - mean;
        variance += centred * centred;
    }
    variance /= static_cast<float>(width);
    const float inverse = 1.0F / std::sqrt(variance + NormEpsilon);
    for (int at = 0; at < width; ++at) {
        output[at] = (input[at] - mean) * inverse * weight[at] + bias[at];
    }
}

// --- WHICH FLOAT HELPERS ARE IN USE ----------------------------------------------------------
//
// The baselines above are the reference: scalar with a 4-wide SSE2 hand-vectorisation where it was
// worth writing. floats::avx2Table() is 8 lanes with FMA, a vectorised exp, and the tanh GELU. The
// choice is one pointer, read once per call site, and it is a PROCESS-WIDE setting rather than a
// per-Segmenter one because the tables are stateless and the alternative is threading a flag through
// every private method.
//
// Off by default. Not because it is slower - it is not - but because the AVX2 GELU is the tanh
// approximation rather than the erf the model trained with, and a numerical change should be asked
// for rather than inherited. useFastFloats() asks for it.
namespace {

void geluBaselineRun(float *values, int count) noexcept {
    for (int at = 0; at < count; ++at) values[at] = geluFast(values[at]);
}

void softmaxFastBaseline(float *values, int count) noexcept {
    softmaxInPlaceBaseline(values, count, true);
}

void addIntoBaseline(float *destination, const float *source, int count) noexcept {
    for (int at = 0; at < count; ++at) destination[at] += source[at];
}

// The baseline: the same loops the model used to run inline, kept as the reference, looped over the
// heads a layer carries.
void attendHeadsBaseline(const float *queries, const float *keys, const float *values, int stride,
                         int rows, int headWidth, int heads, float scale, bool fastSoftmax,
                         float *scores, float *context, int contextStride) noexcept {
    for (int head = 0; head < heads; ++head) {
        const int offset = head * headWidth;
        for (int query = 0; query < rows; ++query) {
            const float *queryVector =
                queries + static_cast<std::size_t>(query) * stride + offset;
            for (int key = 0; key < rows; ++key) {
                scores[key] =
                    dotF32Baseline(queryVector,
                                   keys + static_cast<std::size_t>(key) * stride + offset,
                                   headWidth) *
                    scale;
            }
            softmaxInPlaceBaseline(scores, rows, fastSoftmax);
            float *out = context + static_cast<std::size_t>(query) * contextStride + offset;
            std::memset(out, 0, static_cast<std::size_t>(headWidth) * sizeof(float));
            for (int key = 0; key < rows; ++key) {
                addScaledBaseline(out, values + static_cast<std::size_t>(key) * stride + offset,
                                  scores[key], headWidth);
            }
        }
    }
}

const floats::Table baselineTable{&dotF32Baseline,   &addScaledBaseline,  &addIntoBaseline,
                                  &layerNormBaseline, &geluBaselineRun,   &softmaxFastBaseline,
                                  &attendHeadsBaseline};

std::atomic<const floats::Table *> activeFloats{&baselineTable};

// See ensureFloatDefault below: consumed by whichever of the lazy default or an explicit
// useFastFloats() call happens first, so the default can never overwrite a caller's choice.
std::once_flag floatDefaultOnce;

// WHETHER THE DECISION HEADS RUN EXACTLY while the trunk runs fast. Default OFF, because it was
// measured to buy back NOTHING: over 20,000 names the divergence count is identical with and without,
// at every zero point (sweepsTheFastZeroPoint). The fast kernel's errors enter in the TRUNK's
// representations, and by the time the heads run their inputs have already shifted - an exact head
// faithfully maps a shifted input to the same different answer. The switch stays so the measurement
// remains repeatable.
std::atomic<bool> exactHeads{false};


} // namespace

inline float dotF32(const float *a, const float *b, int count) noexcept {
    return activeFloats.load(std::memory_order_relaxed)->dot(a, b, count);
}

inline void addScaled(float *destination, const float *source, float weight, int count) noexcept {
    activeFloats.load(std::memory_order_relaxed)->addScaled(destination, source, weight, count);
}

inline void addInto(float *destination, const float *source, int count) noexcept {
    activeFloats.load(std::memory_order_relaxed)->addInto(destination, source, count);
}

inline void layerNorm(const float *input, int width, const float *weight, const float *bias,
                      float *output) noexcept {
    activeFloats.load(std::memory_order_relaxed)->layerNorm(input, width, weight, bias, output);
}

inline void geluInPlace(float *values, int count, bool fast) noexcept {
    if (!fast) {
        for (int at = 0; at < count; ++at) values[at] = gelu(values[at]);
        return;
    }
    activeFloats.load(std::memory_order_relaxed)->gelu(values, count);
}

inline void softmaxInPlace(float *values, int count, bool fast = false) noexcept {
    if (!fast) {
        softmaxInPlaceBaseline(values, count, false);
        return;
    }
    activeFloats.load(std::memory_order_relaxed)->softmax(values, count);
}

// One weight matrix, in whichever precisions this run needs. `f32` is filled for tensors stored
// as float (the heads) and, from dequantised codes, when the reference path was requested.
struct Matrix {
    int in = 0;
    int out = 0;
    kernels::PackedMatrix packed;
    std::vector<float> f32; // (out, in) row-major
    std::vector<float> bias;
    // A DECISION HEAD, whose output feeds an argmax rather than another layer. Set for the split, type
    // and global heads at load. When the caller has asked for exact heads, these skip the fast kernel:
    // they are a few per cent of the arithmetic, and a wrong bit here becomes a wrong answer directly
    // instead of being averaged away by the layers above.
    bool decisionHead = false;

    [[nodiscard]] bool quantised() const noexcept { return !packed.codes.empty(); }
};

std::vector<float> dequantise(const Tensor &tensor) {
    if (!tensor.quantised()) {
        return tensor.floats;
    }
    const auto rows = tensor.rows();
    const auto width = tensor.rowWidth();
    std::vector<float> values(static_cast<std::size_t>(tensor.count()));
    for (std::int64_t row = 0; row < rows; ++row) {
        const float scale = tensor.scales[static_cast<std::size_t>(row)];
        for (std::int64_t at = 0; at < width; ++at) {
            values[static_cast<std::size_t>(row * width + at)] =
                static_cast<float>(tensor.codes[static_cast<std::size_t>(row * width + at)]) *
                scale;
        }
    }
    return values;
}

// Per-thread scratch. The parser runs one name at a time per thread; keeping the buffers alive
// across names removes every allocation from the steady state.
struct Scratch {
    std::vector<float> a, b, c, d, e, attention;
    std::vector<std::int8_t> quantised;
    std::vector<std::int16_t> quantised16;
    std::vector<std::int32_t> accumulators;
    std::vector<float> actScales;
    std::vector<float> gather;
    std::vector<std::int8_t> gatherQuantised;
    std::vector<std::int16_t> gatherQuantised16;
};

// NEVER DESTROYED, AND THAT IS THE POINT.
//
// As a plain `thread_local Scratch` this is destroyed when a worker thread exits, and on this
// toolchain that crashes the process: the vectors are allocated through one heap and freed during
// LdrShutdownThread through another, so a pool that finishes a batch and lets its threads go takes
// the process with it. Three separate crashes came from this — a 1.4M-name dump at block two, a
// four-thread batch inside a hundred names, and what looked like concurrent construction.
//
// A leaked instance per parsing thread is a few hundred kilobytes that lives until the process
// exits, which is the right trade for a library that a UI links: the scratch would be reused for
// the whole run anyway, and nothing here owns a resource that matters beyond memory.
thread_local Scratch& scratchStorage = *new Scratch;

thread_local ProfileCounters profileStorage;

// A stopwatch that adds its lifetime to one counter. ~40 ns per section, ~100 sections per name
// — three orders of magnitude below what it measures.
class Timed {
    public:
    explicit Timed(double &sink) noexcept
        : sink_(sink), started_(std::chrono::steady_clock::now()) {}
    ~Timed() {
        sink_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_)
                     .count();
    }
    Timed(const Timed &) = delete;
    Timed &operator=(const Timed &) = delete;

    private:
    double &sink_;
    std::chrono::steady_clock::time_point started_;
};

} // namespace

// Whether the AVX2 float table is in use. Reported as well as set, because a caller that asked for
// it on a CPU without AVX2 has to be told it did not get it rather than left to assume.
bool applyFloatMode(FloatMode mode) noexcept {
    const floats::Table *table = nullptr;
    switch (mode) {
    case FloatMode::Baseline:
        break;
    case FloatMode::Avx2:
        table = floats::avx2Table();
        break;
    case FloatMode::Avx2FastGelu:
        table = floats::avx2FastGeluTable();
        break;
    }
    // AVX2 for the floats is offered only where the kernel dispatcher already proved AVX2 usable on
    // this machine - same CPUID, and the same self-check that demotes a lying SIMD path.
    if (table != nullptr && kernels::activePath() < kernels::Path::Avx2) table = nullptr;
    activeFloats.store(table != nullptr ? table : &baselineTable, std::memory_order_relaxed);
    return table != nullptr;
}

bool useFastFloats(FloatMode mode) noexcept {
    // Consume the once-flag so the lazy default cannot come along later and undo this.
    std::call_once(floatDefaultOnce, [] {});
    return applyFloatMode(mode);
}

bool applyFloatMode(FloatMode mode) noexcept;
void ensureFloatDefault() noexcept;

void useExactHeads(bool wanted) noexcept {
    exactHeads.store(wanted, std::memory_order_relaxed);
}

bool exactHeadsInUse() noexcept { return exactHeads.load(std::memory_order_relaxed); }

const floats::Table *baselineFloats() noexcept { return &baselineTable; }

// THE DEFAULT IS THE BASELINE TABLE - the scalar and SSE2 reference - and that is a deliberate
// reversal.
//
// The AVX2 table is 1.20x faster and its GELU is the same erf the model trained with, accurate to
// about 1e-7, so it is tempting to make it the default. But it changes the parse of 0.3% of names, and
// it cannot not: vectorising a float reduction changes the ORDER of the additions, FMA rounds once
// where a separate multiply and add round twice, and float addition is not associative. There is no
// version of an eight-wide float path that is bit-identical to a four-wide one. Any speed-up that
// changes answers belongs behind a flag rather than in the default, however good its numbers are - the
// shipped default stays reproducible against every earlier build.
//
// Resolved on the first Segmenter rather than at static-init time, because choosing it needs CPUID and
// CPUID lives behind the kernel dispatcher's own lazy resolve.
//
// NEURELEASE_SEGMENTER_FLOATS=avx2 turns it on; =tanhgelu turns it on with the faster, less
// accurate GELU as well. An explicit useFastFloats() call always wins: it consumes the same once-flag,
// so the default can no longer overwrite a caller's choice afterwards.
void ensureFloatDefault() noexcept {
    std::call_once(floatDefaultOnce, [] {
        FloatMode mode = FloatMode::Baseline;
        if (const char *wanted = std::getenv("NEURELEASE_SEGMENTER_FLOATS")) {
            const std::string text(wanted);
            if (text == "avx2") mode = FloatMode::Avx2;
            else if (text == "tanhgelu") mode = FloatMode::Avx2FastGelu;
        }
        applyFloatMode(mode);
    });
}

bool fastFloatsInUse() noexcept {
    return activeFloats.load(std::memory_order_relaxed) != &baselineTable;
}

ProfileCounters profileCounters() noexcept { return profileStorage; }

void resetProfileCounters() noexcept { profileStorage = {}; }

struct Segmenter::Prepared {
    Precision precision = Precision::Int8;

    // Embeddings, dequantised: a lookup is a row copy, quantising it would only add error.
    std::vector<float> charTable, scriptTable, caseTable;
    int charWidth = 0, scriptWidth = 0, caseWidth = 0;
    int charRows = 0, scriptRows = 0, caseRows = 0;
    int embedding = 0;

    struct Conv {
        Matrix weights; // width = kernel * in, laid out (out, kernel, in)
        int in = 0, out = 0, kernel = 1, dilation = 1;
        std::vector<float> normWeight, normBias;
    };
    std::vector<Conv> convs;
    int charChannels = 0;

    Matrix project;
    std::vector<float> position, positionEnd, summary;
    int longest = 0, summaries = 0, tokenWidth = 0, heads = 0;

    struct EncoderLayer {
        Matrix inProj, outProj, linear1, linear2;
        std::vector<float> norm1Weight, norm1Bias, norm2Weight, norm2Bias;
    };
    std::vector<EncoderLayer> layers;

    bool fastActivation = false;

    [[nodiscard]] float activate(float value) const noexcept {
        return fastActivation ? geluFast(value) : gelu(value);
    }

    Matrix splitHidden, splitOut, typeHead, globalTrunk;
    std::vector<Matrix> globalHeads;

    std::vector<std::string> spanTypes;
    std::vector<GlobalField> globalFields;

    // --- forward-pass pieces, defined in this translation unit ---
    void applyMatrix(const Matrix &matrix, const float *input, int rows, float *output,
                     bool geluAfter = false) const;
    void applyConv(const Conv &conv, const float *input, int length, float *output) const;
    void attention(const EncoderLayer &layer, int rows, std::vector<float> &state) const;
};

namespace {

Matrix makeMatrix(const Tensor &weight, const Tensor *bias, Precision precision) {
    Matrix matrix;
    matrix.out = static_cast<int>(weight.rows());
    matrix.in = static_cast<int>(weight.rowWidth());
    if (weight.quantised()) {
        matrix.packed.pack(weight.codes.data(), weight.scales.data(), matrix.out, matrix.in);
        if (precision == Precision::Float32Reference) {
            matrix.f32 = dequantise(weight);
        }
    } else {
        matrix.f32 = weight.floats;
    }
    if (bias != nullptr) {
        matrix.bias = bias->floats;
    }
    return matrix;
}

// Convolution weights arrive as (out, in, kernel); the kernels want each output row contiguous in
// the order the activation window is laid out, which is (kernel, in) — position-major.
Matrix makeConvMatrix(const Tensor &weight, const Tensor &bias, Precision precision) {
    const int out = static_cast<int>(weight.shape[0]);
    const int in = static_cast<int>(weight.shape[1]);
    const int kernel = static_cast<int>(weight.shape[2]);
    Matrix matrix;
    matrix.out = out;
    matrix.in = in * kernel;
    matrix.bias = bias.floats;

    const auto reorder = [&](const auto *source, auto *destination) {
        for (int o = 0; o < out; ++o) {
            for (int k = 0; k < kernel; ++k) {
                for (int c = 0; c < in; ++c) {
                    destination[(static_cast<std::size_t>(o) * kernel + k) * in + c] =
                        source[(static_cast<std::size_t>(o) * in + c) * kernel + k];
                }
            }
        }
    };
    if (weight.quantised()) {
        std::vector<std::int8_t> reordered(weight.codes.size());
        reorder(weight.codes.data(), reordered.data());
        matrix.packed.pack(reordered.data(), weight.scales.data(), out, in * kernel);
        if (precision == Precision::Float32Reference) {
            const auto dense = dequantise(weight);
            matrix.f32.resize(dense.size());
            reorder(dense.data(), matrix.f32.data());
        }
    } else {
        matrix.f32.resize(weight.floats.size());
        reorder(weight.floats.data(), matrix.f32.data());
    }
    return matrix;
}

std::vector<std::string> splitLines(const std::string &text) {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

} // namespace

// `geluAfter` fuses the activation into the dequant loop, PER ROW. True whole-loop fusion would
// bake GELU into kernels::finish, but the same cache effect is had far more simply: finish writes a
// row (4 KB at the FFN's width), and applying GELU to that row immediately - while it is still in
// L1 - costs a register pass instead of the whole-tensor read+write the separate geluInPlace() pass
// paid after ALL rows were written and the first ones long evicted. The unfused shape walked the
// N x 1024 hidden tensor three times per layer (write, gelu, quantise-read); this makes it two.
void Segmenter::Prepared::applyMatrix(const Matrix &matrix, const float *input, int rows,
                                      float *output, bool geluAfter) const {
    Timed timer(profileStorage.matmulSeconds);
    const bool useInt8 = precision == Precision::Int8 && matrix.quantised();
    if (useInt8) {
        auto &scratch = scratchStorage;
        const int stride = matrix.packed.stride;
        // A decision head takes the exact kernel when asked to, which means it also needs the exact
        // kernel's activation width - the two go together, and getting them out of step would feed
        // int8 codes to a kernel reading int16.
        const bool exact = matrix.decisionHead && exactHeads.load(std::memory_order_relaxed);
        const auto matmul = exact ? kernels::exactMatmul() : nullptr;
        const bool wide = (exact ? kernels::exactActivationKind() : kernels::activationKind()) ==
                          kernels::ActivationKind::Int16;
        scratch.actScales.resize(static_cast<std::size_t>(rows));
        scratch.accumulators.resize(static_cast<std::size_t>(rows) * matrix.out);
        // One activation scale PER ROW (per token), not per map: a pooled token vector and an
        // attention context vary a lot in magnitude between rows, and a shared scale hands the
        // quietest row the fewest levels. All rows are quantised first and pushed through ONE
        // batched matmul, so the kernel reuses each loaded weight row across the whole batch.
        if (wide) {
            scratch.quantised16.resize(static_cast<std::size_t>(rows) * stride);
            for (int row = 0; row < rows; ++row) {
                scratch.actScales[static_cast<std::size_t>(row)] = kernels::quantise16(
                    input + static_cast<std::size_t>(row) * matrix.in, matrix.in,
                    scratch.quantised16.data() + static_cast<std::size_t>(row) * stride, stride);
            }
            if (matmul != nullptr)
                matmul(matrix.packed, scratch.quantised16.data(), rows, stride,
                       scratch.accumulators.data());
            else
                kernels::matmul(matrix.packed, scratch.quantised16.data(), rows, stride,
                                scratch.accumulators.data());
        } else {
            scratch.quantised.resize(static_cast<std::size_t>(rows) * stride);
            for (int row = 0; row < rows; ++row) {
                scratch.actScales[static_cast<std::size_t>(row)] = kernels::quantise(
                    input + static_cast<std::size_t>(row) * matrix.in, matrix.in,
                    scratch.quantised.data() + static_cast<std::size_t>(row) * stride, stride);
            }
            if (matmul != nullptr)
                matmul(matrix.packed, scratch.quantised.data(), rows, stride,
                       scratch.accumulators.data());
            else
                kernels::matmul(matrix.packed, scratch.quantised.data(), rows, stride,
                                scratch.accumulators.data());
        }
        for (int row = 0; row < rows; ++row) {
            float *destination = output + static_cast<std::size_t>(row) * matrix.out;
            kernels::finish(matrix.packed,
                            scratch.accumulators.data() + static_cast<std::size_t>(row) * matrix.out,
                            scratch.actScales[static_cast<std::size_t>(row)],
                            matrix.bias.empty() ? nullptr : matrix.bias.data(), destination);
            if (geluAfter) geluInPlace(destination, matrix.out, fastActivation);
        }
        return;
    }
    for (int row = 0; row < rows; ++row) {
        const float *source = input + static_cast<std::size_t>(row) * matrix.in;
        float *destination = output + static_cast<std::size_t>(row) * matrix.out;
        for (int o = 0; o < matrix.out; ++o) {
            const float *weights = matrix.f32.data() + static_cast<std::size_t>(o) * matrix.in;
            const float base = matrix.bias.empty() ? 0.0F : matrix.bias[static_cast<std::size_t>(o)];
            destination[o] = base + dotF32(weights, source, matrix.in);
        }
        if (geluAfter) geluInPlace(destination, matrix.out, fastActivation);
    }
}

void Segmenter::Prepared::applyConv(const Conv &conv, const float *input, int length,
                                    float *output) const {
    Timed timer(profileStorage.convolutionSeconds);
    auto &scratch = scratchStorage;
    const int pad = conv.dilation * (conv.kernel / 2);
    const int paddedRows = length + 2 * pad;
    const int in = conv.in;
    const int window = conv.kernel * in;

    const bool useInt8 = precision == Precision::Int8 && conv.weights.quantised();
    if (useInt8) {
        const int stride = conv.weights.packed.stride;
        const int total = length * in;
        scratch.accumulators.resize(static_cast<std::size_t>(length) * conv.out);

        // The whole padded map is quantised once under one scale, and every position goes through
        // ONE batched matmul. A dilation-1 convolution needs no im2col at all: its windows are
        // overlapping views into the map, so the activation stride is just the channel count and
        // the kernel's over-read past each window multiplies against zero weight padding (the map
        // carries a 64-element zero tail so the read stays in bounds). Dilated convolutions
        // gather their windows into contiguous rows first — im2col, kept off the fast path.
        const auto sweep = [&](auto *map, auto &gathered, float scale) {
            using Code = std::remove_reference_t<decltype(*map)>;
            if (conv.dilation == 1) {
                kernels::matmul(conv.weights.packed, map, length, in,
                                scratch.accumulators.data());
            } else {
                gathered.assign(static_cast<std::size_t>(length) * stride, 0);
                for (int position = 0; position < length; ++position) {
                    Code *row = gathered.data() + static_cast<std::size_t>(position) * stride;
                    for (int k = 0; k < conv.kernel; ++k) {
                        std::memcpy(row + static_cast<std::size_t>(k) * in,
                                    map + (static_cast<std::size_t>(position) +
                                           static_cast<std::size_t>(k) * conv.dilation) *
                                              in,
                                    static_cast<std::size_t>(in) * sizeof(Code));
                    }
                }
                kernels::matmul(conv.weights.packed, gathered.data(), length, stride,
                                scratch.accumulators.data());
            }
            for (int position = 0; position < length; ++position) {
                kernels::finish(
                    conv.weights.packed,
                    scratch.accumulators.data() + static_cast<std::size_t>(position) * conv.out,
                    scale, conv.weights.bias.data(),
                    output + static_cast<std::size_t>(position) * conv.out);
            }
        };
        if (kernels::activationKind() == kernels::ActivationKind::Int16) {
            scratch.quantised16.assign(static_cast<std::size_t>(paddedRows) * in + 64, 0);
            const float scale = kernels::quantise16(
                input, total, scratch.quantised16.data() + static_cast<std::size_t>(pad) * in,
                total);
            sweep(scratch.quantised16.data(), scratch.gatherQuantised16, scale);
        } else {
            scratch.quantised.assign(static_cast<std::size_t>(paddedRows) * in + 64, 0);
            const float scale = kernels::quantise(
                input, total, scratch.quantised.data() + static_cast<std::size_t>(pad) * in,
                total);
            sweep(scratch.quantised.data(), scratch.gatherQuantised, scale);
        }
    } else {
        scratch.gather.assign(static_cast<std::size_t>(paddedRows) * in, 0.0F);
        std::memcpy(scratch.gather.data() + static_cast<std::size_t>(pad) * in, input,
                    static_cast<std::size_t>(length) * in * sizeof(float));
        for (int position = 0; position < length; ++position) {
            float *destination = output + static_cast<std::size_t>(position) * conv.out;
            for (int o = 0; o < conv.out; ++o) {
                const float *weights =
                    conv.weights.f32.data() + static_cast<std::size_t>(o) * window;
                float sum = conv.weights.bias[static_cast<std::size_t>(o)];
                for (int k = 0; k < conv.kernel; ++k) {
                    const float *source =
                        scratch.gather.data() +
                        (static_cast<std::size_t>(position) +
                         static_cast<std::size_t>(k) * conv.dilation) *
                            in;
                    const float *kernelWeights = weights + static_cast<std::size_t>(k) * in;
                    for (int c = 0; c < in; ++c) {
                        sum += kernelWeights[c] * source[c];
                    }
                }
                destination[o] = sum;
            }
        }
    }

    for (int position = 0; position < length; ++position) {
        float *row = output + static_cast<std::size_t>(position) * conv.out;
        layerNorm(row, conv.out, conv.normWeight.data(), conv.normBias.data(), row);
        geluInPlace(row, conv.out, fastActivation);
    }
}

void Segmenter::Prepared::attention(const EncoderLayer &layer, int rows,
                                    std::vector<float> &state) const {
    auto &scratch = scratchStorage;
    const int width = tokenWidth;
    const int headWidth = width / heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(headWidth));

    // Pre-LN: state += outProj(attention(norm1(state))), then state += ffn(norm2(state)).
    scratch.b.resize(static_cast<std::size_t>(rows) * width);
    for (int row = 0; row < rows; ++row) {
        layerNorm(state.data() + static_cast<std::size_t>(row) * width, width,
                  layer.norm1Weight.data(), layer.norm1Bias.data(),
                  scratch.b.data() + static_cast<std::size_t>(row) * width);
    }
    scratch.c.resize(static_cast<std::size_t>(rows) * 3 * width); // q, k, v per row
    applyMatrix(layer.inProj, scratch.b.data(), rows, scratch.c.data());

    scratch.d.resize(static_cast<std::size_t>(rows) * width); // context, head-concatenated
    scratch.attention.resize(static_cast<std::size_t>(rows));
    const auto attentionStarted = std::chrono::steady_clock::now();

    // One call per head. The projection wrote q, k and v interleaved, so all three live in the same
    // buffer 3*width floats apart per token, and the kernel is told that stride rather than being
    // handed three repacked copies. It does its own packing internally, where it can pack into the
    // exact layout its inner loops want.
    // ONE CALL FOR THE WHOLE LAYER. It was one call per head, and the per-call setup - transpose,
    // packing, scratch bookkeeping, the indirection itself - measured at 96% of a short name's
    // attention cost, paid heads-times-layers times over. The kernel now amortises all of it across
    // the layer's heads; see AttendFunction in SegmenterFloats.h for the numbers.
    const floats::Table *table = activeFloats.load(std::memory_order_relaxed);
    table->attendHeads(scratch.c.data(), scratch.c.data() + width, scratch.c.data() + 2 * width,
                       3 * width, rows, headWidth, heads, scale, fastActivation,
                       scratch.attention.data(), scratch.d.data(), width);
    profileStorage.attentionCalls += heads;
    profileStorage.attentionSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - attentionStarted)
            .count();
    scratch.e.resize(static_cast<std::size_t>(rows) * width);
    applyMatrix(layer.outProj, scratch.d.data(), rows, scratch.e.data());
    addInto(state.data(), scratch.e.data(), static_cast<int>(state.size()));

    for (int row = 0; row < rows; ++row) {
        layerNorm(state.data() + static_cast<std::size_t>(row) * width, width,
                  layer.norm2Weight.data(), layer.norm2Bias.data(),
                  scratch.b.data() + static_cast<std::size_t>(row) * width);
    }
    const int feedforward = layer.linear1.out;
    scratch.c.resize(static_cast<std::size_t>(rows) * feedforward);
    // GELU fused into the dequant loop: the hidden tensor is the widest thing in the model
    // (rows x 1024), and activating it row-by-row while each row is still in L1 removes a whole
    // read+write pass over it.
    applyMatrix(layer.linear1, scratch.b.data(), rows, scratch.c.data(), /*geluAfter=*/true);
    applyMatrix(layer.linear2, scratch.c.data(), rows, scratch.e.data());
    addInto(state.data(), scratch.e.data(), static_cast<int>(state.size()));
}

Segmenter::Segmenter(const Weights &weights, Precision precision) {
    ensureFloatDefault();
    auto prepared = std::make_shared<Prepared>();
    prepared->precision = precision;
    prepared->fastActivation = precision == Precision::Int8;

    const auto require = [&](std::string_view name) -> const Tensor & {
        return weights.tensor(name);
    };

    if (weights.scalar("meta.separator_tokens") != 0.0F ||
        weights.scalar("meta.logograph_tokens") != 0.0F) {
        throw std::runtime_error(
            "segmenter: separator/logograph token variants are not implemented in C++");
    }
    if (weights.scalar("meta.token_arch") != 1.0F) {
        throw std::runtime_error(
            "segmenter: only the transformer token stage is implemented in C++");
    }

    const auto &charTable = require("encoder.chars.weight");
    const auto &scriptTable = require("encoder.script.weight");
    const auto &caseTable = require("encoder.case.weight");
    prepared->charTable = dequantise(charTable);
    prepared->scriptTable = dequantise(scriptTable);
    prepared->caseTable = dequantise(caseTable);
    prepared->charRows = static_cast<int>(charTable.rows());
    prepared->scriptRows = static_cast<int>(scriptTable.rows());
    prepared->caseRows = static_cast<int>(caseTable.rows());
    prepared->charWidth = static_cast<int>(charTable.rowWidth());
    prepared->scriptWidth = static_cast<int>(scriptTable.rowWidth());
    prepared->caseWidth = static_cast<int>(caseTable.rowWidth());
    prepared->embedding = prepared->charWidth + prepared->scriptWidth + prepared->caseWidth;

    const auto &dilations = require("meta.char_dilations").floats;
    for (int index = 0, conv = 0; index < 64; ++index) {
        const std::string key = "char_trunk." + std::to_string(index) + ".weight";
        if (!weights.has(key)) {
            continue;
        }
        Prepared::Conv layer;
        const auto &weight = weights.tensor(key);
        if (weight.shape.size() != 3) {
            continue; // a norm scale, matched by the .norm keys below
        }
        layer.weights = makeConvMatrix(
            weight, require("char_trunk." + std::to_string(index) + ".bias"), precision);
        layer.out = static_cast<int>(weight.shape[0]);
        layer.in = static_cast<int>(weight.shape[1]);
        layer.kernel = static_cast<int>(weight.shape[2]);
        if (conv >= static_cast<int>(dilations.size())) {
            throw std::runtime_error("segmenter: more convolutions than meta.char_dilations");
        }
        layer.dilation = static_cast<int>(dilations[static_cast<std::size_t>(conv)]);
        const std::string norm = "char_trunk." + std::to_string(index + 1) + ".norm.";
        layer.normWeight = require(norm + "weight").floats;
        layer.normBias = require(norm + "bias").floats;
        prepared->convs.push_back(std::move(layer));
        ++conv;
    }
    if (prepared->convs.empty()) {
        throw std::runtime_error("segmenter: no character convolutions in the artifact");
    }
    if (prepared->convs.front().in != prepared->embedding) {
        throw std::runtime_error("segmenter: first convolution width does not match embeddings");
    }
    prepared->charChannels = prepared->convs.back().out;

    prepared->project =
        makeMatrix(require("token_trunk.project.weight"),
                   &require("token_trunk.project.bias"), precision);
    prepared->tokenWidth = prepared->project.out;
    prepared->position = dequantise(require("token_trunk.position.weight"));
    prepared->positionEnd = dequantise(require("token_trunk.position_end.weight"));
    prepared->longest = static_cast<int>(require("token_trunk.position.weight").rows());
    prepared->summary = dequantise(require("token_trunk.summary"));
    prepared->summaries = static_cast<int>(require("token_trunk.summary").rows());
    prepared->heads = static_cast<int>(weights.scalar("meta.token_heads"));
    if (prepared->heads <= 0 || prepared->tokenWidth % prepared->heads != 0) {
        throw std::runtime_error("segmenter: token width is not divisible by the head count");
    }

    for (int index = 0;; ++index) {
        const std::string base = "token_trunk.encoder.layers." + std::to_string(index) + ".";
        if (!weights.has(base + "self_attn.in_proj_weight")) {
            break;
        }
        Prepared::EncoderLayer layer;
        layer.inProj = makeMatrix(require(base + "self_attn.in_proj_weight"),
                                  &require(base + "self_attn.in_proj_bias"), precision);
        layer.outProj = makeMatrix(require(base + "self_attn.out_proj.weight"),
                                   &require(base + "self_attn.out_proj.bias"), precision);
        layer.linear1 = makeMatrix(require(base + "linear1.weight"),
                                   &require(base + "linear1.bias"), precision);
        layer.linear2 = makeMatrix(require(base + "linear2.weight"),
                                   &require(base + "linear2.bias"), precision);
        layer.norm1Weight = require(base + "norm1.weight").floats;
        layer.norm1Bias = require(base + "norm1.bias").floats;
        layer.norm2Weight = require(base + "norm2.weight").floats;
        layer.norm2Bias = require(base + "norm2.bias").floats;
        prepared->layers.push_back(std::move(layer));
    }
    if (prepared->layers.empty()) {
        throw std::runtime_error("segmenter: no transformer layers in the artifact");
    }

    prepared->splitHidden =
        makeMatrix(require("split_head.0.weight"), &require("split_head.0.bias"), precision);
    prepared->splitOut =
        makeMatrix(require("split_head.2.weight"), &require("split_head.2.bias"), precision);
    prepared->typeHead =
        makeMatrix(require("type_head.weight"), &require("type_head.bias"), precision);
    prepared->globalTrunk = makeMatrix(require("global_trunk.0.weight"),
                                       &require("global_trunk.0.bias"), precision);

    prepared->spanTypes = splitLines(weights.text("meta.span_types"));
    // The heads, flagged so applyMatrix can keep them exact. Everything else is trunk.
    prepared->splitHidden.decisionHead = true;
    prepared->splitOut.decisionHead = true;
    prepared->typeHead.decisionHead = true;
    prepared->globalTrunk.decisionHead = true;

    if (static_cast<int>(prepared->spanTypes.size()) != prepared->typeHead.out) {
        throw std::runtime_error("segmenter: span type names do not match the type head");
    }
    for (const auto &line : splitLines(weights.text("meta.global_fields"))) {
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("segmenter: malformed meta.global_fields line: " + line);
        }
        GlobalField field;
        field.name = line.substr(0, equals);
        std::string values = line.substr(equals + 1);
        std::size_t start = 0;
        while (start <= values.size()) {
            const auto comma = values.find(',', start);
            field.values.push_back(values.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start));
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        prepared->globalFields.push_back(std::move(field));
    }
    if (static_cast<int>(prepared->globalFields.size()) != prepared->summaries) {
        throw std::runtime_error("segmenter: global fields do not match the summary tokens");
    }
    for (const auto &field : prepared->globalFields) {
        const std::string base = "global_heads." + field.name + ".";
        Matrix head = makeMatrix(require(base + "weight"), &require(base + "bias"), precision);
        if (head.out != static_cast<int>(field.values.size())) {
            throw std::runtime_error("segmenter: head size mismatch for " + field.name);
        }
        head.decisionHead = true;
        prepared->globalHeads.push_back(std::move(head));
    }

    if (precision == Precision::Int8) {
        // Touch the dispatcher now so the first parse does not pay CPUID resolution.
        (void)kernels::activePath();
    }
    prepared_ = std::move(prepared);
}

void Segmenter::characterStage(const EncodedName &name, std::vector<float> &hidden) const {
    const auto &prepared = *prepared_;
    const int length = static_cast<int>(name.chars.size());
    auto &scratch = scratchStorage;

    scratch.a.resize(static_cast<std::size_t>(length) * prepared.embedding);
    for (int position = 0; position < length; ++position) {
        float *row = scratch.a.data() + static_cast<std::size_t>(position) * prepared.embedding;
        const int charId = std::min<int>(name.chars[static_cast<std::size_t>(position)],
                                         prepared.charRows - 1);
        const int scriptId = std::min<int>(name.script[static_cast<std::size_t>(position)],
                                           prepared.scriptRows - 1);
        const int caseId = std::min<int>(name.caseFlags[static_cast<std::size_t>(position)],
                                         prepared.caseRows - 1);
        std::memcpy(row,
                    prepared.charTable.data() +
                        static_cast<std::size_t>(charId) * prepared.charWidth,
                    static_cast<std::size_t>(prepared.charWidth) * sizeof(float));
        std::memcpy(row + prepared.charWidth,
                    prepared.scriptTable.data() +
                        static_cast<std::size_t>(scriptId) * prepared.scriptWidth,
                    static_cast<std::size_t>(prepared.scriptWidth) * sizeof(float));
        std::memcpy(row + prepared.charWidth + prepared.scriptWidth,
                    prepared.caseTable.data() +
                        static_cast<std::size_t>(caseId) * prepared.caseWidth,
                    static_cast<std::size_t>(prepared.caseWidth) * sizeof(float));
    }

    // Ping-pong between two buffers; `hidden` receives the last layer's output directly.
    std::vector<float> *input = &scratch.a;
    std::vector<float> *spare = &scratch.b;
    for (std::size_t index = 0; index < prepared.convs.size(); ++index) {
        const auto &conv = prepared.convs[index];
        const bool last = index + 1 == prepared.convs.size();
        std::vector<float> *destination = last ? &hidden : spare;
        destination->resize(static_cast<std::size_t>(length) * conv.out);
        prepared.applyConv(conv, input->data(), length, destination->data());
        spare = input;
        input = destination;
    }
}

void Segmenter::pool(const EncodedName &name, const std::vector<float> &hidden,
                     std::vector<float> &pooled) const {
    const auto &prepared = *prepared_;
    const int channels = prepared.charChannels;
    const int tokens = std::max<int>(static_cast<int>(name.firstPosition.size()), 1);
    pooled.assign(static_cast<std::size_t>(tokens) * channels,
                  -std::numeric_limits<float>::infinity());
    const int length = static_cast<int>(name.tokenOf.size());
    for (int position = 0; position < length; ++position) {
        const auto token = name.tokenOf[static_cast<std::size_t>(position)];
        if (token < 0 || token >= tokens) {
            continue;
        }
        float *destination = pooled.data() + static_cast<std::size_t>(token) * channels;
        const float *source = hidden.data() + static_cast<std::size_t>(position) * channels;
        for (int c = 0; c < channels; ++c) {
            destination[c] = std::max(destination[c], source[c]);
        }
    }
    for (auto &value : pooled) {
        if (value == -std::numeric_limits<float>::infinity()) {
            value = 0.0F;
        }
    }
}

void Segmenter::tokenStage(const EncodedName &name, const std::vector<float> &pooled,
                           std::vector<float> &tokenHidden, std::vector<float> &summary) const {
    const auto &prepared = *prepared_;
    const int width = prepared.tokenWidth;
    const int tokens = static_cast<int>(pooled.size()) / prepared.charChannels;
    const int rows = tokens + prepared.summaries;
    auto &scratch = scratchStorage;

    scratch.a.resize(static_cast<std::size_t>(tokens) * width);
    prepared.applyMatrix(prepared.project, pooled.data(), tokens, scratch.a.data());

    std::vector<float> state(static_cast<std::size_t>(rows) * width);
    std::memcpy(state.data(), prepared.summary.data(),
                static_cast<std::size_t>(prepared.summaries) * width * sizeof(float));
    for (int token = 0; token < tokens; ++token) {
        float *row = state.data() +
                     (static_cast<std::size_t>(prepared.summaries) + token) * width;
        const int fromStart = std::min(token, prepared.longest - 1);
        const int fromEnd = std::clamp(tokens - 1 - token, 0, prepared.longest - 1);
        const float *projected = scratch.a.data() + static_cast<std::size_t>(token) * width;
        const float *position =
            prepared.position.data() + static_cast<std::size_t>(fromStart) * width;
        const float *positionEnd =
            prepared.positionEnd.data() + static_cast<std::size_t>(fromEnd) * width;
        for (int at = 0; at < width; ++at) {
            row[at] = projected[at] + position[at] + positionEnd[at];
        }
    }

    for (const auto &layer : prepared.layers) {
        prepared.attention(layer, rows, state);
    }

    summary.assign(state.begin(),
                   state.begin() + static_cast<std::size_t>(prepared.summaries) * width);
    tokenHidden.assign(state.begin() + static_cast<std::size_t>(prepared.summaries) * width,
                       state.end());
}

Prediction Segmenter::run(const EncodedName &name) const {
    Timed timer(profileStorage.totalSeconds);
    ++profileStorage.names;
    const auto &prepared = *prepared_;
    Prediction prediction;

    using Clock = std::chrono::steady_clock;
    const auto micros = [](Clock::time_point from, Clock::time_point to) {
        return std::chrono::duration<float, std::micro>(to - from).count();
    };
    const auto started = Clock::now();

    std::vector<float> hidden, pooled, tokenHidden, summary;
    characterStage(name, hidden);
    const auto characterDone = Clock::now();
    pool(name, hidden, pooled);
    tokenStage(name, pooled, tokenHidden, summary);
    const auto tokenDone = Clock::now();

    const int runCount = static_cast<int>(name.firstPosition.size());
    const int charChannels = prepared.charChannels;
    const int width = prepared.tokenWidth;

    // THE HEADS RUN AS ONE BATCH PER MATRIX, not once per token.
    //
    // This loop used to call applyMatrix with rows = 1, three times for every run in the name — a
    // 40-token name meant 120 matmuls, each one loading the whole weight matrix from memory to
    // multiply a single row against it. The weights are the expensive thing to move: one call with
    // rows = runCount loads each weight row once and reuses it across every token, which is what
    // the kernels are built to do (see applyMatrix: rows are quantised individually, then pushed
    // through ONE kernels::matmul).
    //
    // NUMERICALLY IDENTICAL, and that is not an accident: applyMatrix computes one activation
    // scale PER ROW, so a row's quantisation does not depend on what it is batched with. Batching
    // changes how many times the weights are walked and nothing else — segmenter_model_tests pins
    // this against the Python reference bit for bit.
    prediction.splitStart.resize(static_cast<std::size_t>(runCount));
    prediction.typeScores.resize(static_cast<std::size_t>(runCount));
    if (runCount > 0) {
        const int jointWidth = charChannels + width;
        const int splitHiddenOut = prepared.splitHidden.out;
        const int typeOut = prepared.typeHead.out;

        std::vector<float> joint(static_cast<std::size_t>(runCount) * jointWidth);
        for (int run = 0; run < runCount; ++run) {
            const auto first = name.firstPosition[static_cast<std::size_t>(run)];
            float *row = joint.data() + static_cast<std::size_t>(run) * jointWidth;
            std::memcpy(row, hidden.data() + static_cast<std::size_t>(first) * charChannels,
                        static_cast<std::size_t>(charChannels) * sizeof(float));
            std::memcpy(row + charChannels,
                        tokenHidden.data() + static_cast<std::size_t>(run) * width,
                        static_cast<std::size_t>(width) * sizeof(float));
        }

        std::vector<float> splitHidden(static_cast<std::size_t>(runCount) * splitHiddenOut);
        prepared.applyMatrix(prepared.splitHidden, joint.data(), runCount, splitHidden.data());
        for (auto &value : splitHidden) {
            value = prepared.activate(value);
        }

        std::vector<float> splitLogits(static_cast<std::size_t>(runCount) * 2);
        prepared.applyMatrix(prepared.splitOut, splitHidden.data(), runCount, splitLogits.data());

        // The type head reads the token hidden states directly, which are already contiguous rows.
        std::vector<float> typeScores(static_cast<std::size_t>(runCount) * typeOut);
        prepared.applyMatrix(prepared.typeHead, tokenHidden.data(), runCount, typeScores.data());

        // Softmax stays per row: it is a normalisation WITHIN one token's logits, and batching it
        // across tokens would be a different function.
        for (int run = 0; run < runCount; ++run) {
            float *logits = splitLogits.data() + static_cast<std::size_t>(run) * 2;
            softmaxInPlace(logits, 2);
            prediction.splitStart[static_cast<std::size_t>(run)] = logits[1];

            auto &scores = prediction.typeScores[static_cast<std::size_t>(run)];
            scores.assign(typeScores.begin() + static_cast<std::size_t>(run) * typeOut,
                          typeScores.begin() + static_cast<std::size_t>(run + 1) * typeOut);
            softmaxInPlace(scores.data(), typeOut);
        }
    }

    // The whole-name heads batch the same way through their shared trunk. Only the trunk: each
    // field has its OWN output matrix after it, and different matrices cannot share a call.
    prediction.globals.resize(static_cast<std::size_t>(prepared.summaries));
    if (prepared.summaries > 0) {
        const int trunkOutWidth = prepared.globalTrunk.out;
        std::vector<float> trunkOut(static_cast<std::size_t>(prepared.summaries) * trunkOutWidth);
        prepared.applyMatrix(prepared.globalTrunk, summary.data(), prepared.summaries,
                            trunkOut.data());
        for (auto &value : trunkOut) {
            value = prepared.activate(value);
        }
        for (int field = 0; field < prepared.summaries; ++field) {
            auto &out = prediction.globals[static_cast<std::size_t>(field)];
            const auto &head = prepared.globalHeads[static_cast<std::size_t>(field)];
            out.resize(static_cast<std::size_t>(head.out));
            prepared.applyMatrix(head,
                                 trunkOut.data() + static_cast<std::size_t>(field) * trunkOutWidth,
                                 1, out.data());
            softmaxInPlace(out.data(), head.out);
        }
    }

    const auto finished = Clock::now();
    const auto status = kernels::pathStatus();
    prediction.info.kernelPath = kernels::pathName(status.active);
    prediction.info.kernelDemoted = status.demoted;
    prediction.info.characterMicros = micros(started, characterDone);
    prediction.info.tokenMicros = micros(characterDone, tokenDone);
    prediction.info.totalMicros = micros(started, finished);
    return prediction;
}

std::vector<PredictedSpan> Segmenter::spans(const EncodedName &name,
                                            const Prediction &prediction) {
    std::vector<PredictedSpan> out;
    const auto runCount = name.runSpans.size();
    if (runCount == 0 || prediction.typeScores.size() < runCount) {
        return out;
    }
    const auto typeCount = prediction.typeScores.front().size();
    std::vector<float> mean(typeCount);

    std::size_t begin = 0;
    while (begin < runCount) {
        std::size_t end = begin + 1;
        while (end < runCount && prediction.splitStart[end] <= 0.5F) {
            ++end;
        }
        std::fill(mean.begin(), mean.end(), 0.0F);
        for (std::size_t run = begin; run < end; ++run) {
            for (std::size_t type = 0; type < typeCount; ++type) {
                mean[type] += prediction.typeScores[run][type];
            }
        }
        const auto best = static_cast<int>(
            std::max_element(mean.begin(), mean.end()) - mean.begin());
        if (best != 0) { // 0 is "nothing here"
            out.push_back({best, name.runSpans[begin].first, name.runSpans[end - 1].second,
                           mean[static_cast<std::size_t>(best)] /
                               static_cast<float>(end - begin)});
        }
        begin = end;
    }
    return out;
}

const std::vector<std::string> &Segmenter::spanTypes() const noexcept {
    return prepared_->spanTypes;
}

const std::vector<Segmenter::GlobalField> &Segmenter::globalFields() const noexcept {
    return prepared_->globalFields;
}

Precision Segmenter::precision() const noexcept { return prepared_->precision; }

int Segmenter::charChannels() const noexcept { return prepared_->charChannels; }

int Segmenter::tokenChannels() const noexcept { return prepared_->tokenWidth; }

} // namespace neurelease::model
