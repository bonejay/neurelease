// The int8 inner loops, behind one function and a runtime CPU check.
//
// Everything expensive in the segmenter is a matrix–vector product against a quantised weight
// matrix, so that is the ONLY thing implemented per instruction set. The layers around it —
// normalisation, GELU, softmax, attention scores — are ~1% of the multiply–accumulates and stay
// in shared float32 code, which also keeps every kernel path bit-identical: a kernel produces raw
// int32 dot products, and the one shared epilogue turns them into floats. Integer addition is
// associative, so scalar, AVX2 and VNNI must agree EXACTLY, and the tests assert that they do.
//
// Signedness is the trap in x86 int8: `vpmaddubsw` and `vpdpbusd` multiply UNSIGNED bytes by
// signed bytes. Activations are signed, so the VNNI paths compute with a + 128 and subtract the
// precomputed correction 128 * sum(row) afterwards — that sum lives in PackedMatrix. The AVX2
// path avoids the trap (and `vpmaddubsw`'s int16 saturation: 255*127*2 overflows 32767) with
// `vpmaddwd`, which is signed and exact over int16 operands.
//
// Each path also declares how it wants the activation delivered. AVX2 has no int8 dot product,
// so widening the activation to int16 ONCE — instead of re-widening it for every weight row —
// halves the instruction count of its inner loop. The codes are identical either way, so paths
// still agree bit for bit.

#pragma once

#include <cstdint>
#include <vector>

namespace neurelease::model::kernels {

enum class Path : std::uint8_t { Scalar, Avx2, Avx2Vnni, Avx512Vnni };

const char *pathName(Path path) noexcept;

// The best path this CPU supports, resolved once via CPUID (and XGETBV, because AVX-512 state
// must also be OS-enabled). The environment variable NEURELEASE_SEGMENTER_KERNEL, when set to
// one of the path names ("scalar", "avx2", "avx2vnni", "avx512vnni"), caps the choice — a field
// diagnostic, not an API.
Path activePath() noexcept;

// Force a specific path for tests and benchmarks. Throws std::runtime_error if the CPU does not
// support it. Not thread-safe against concurrent matvec calls; call it during setup.
void forcePath(Path path);

// --- THE UNSIGNED-ACTIVATION EXPERIMENT ---------------------------------------------------
//
// The fast u8 x s8 instruction (VPMADDUBSW) multiplies an UNSIGNED byte by a SIGNED byte and adds
// ADJACENT PAIRS into a signed 16-bit lane — which saturates: 255 * 127 * 2 is 64,770 and int16
// stops at 32,767. FBGEMM uses it anyway and documents the saturation as acceptable. This model
// uses signed activations and VPMADDWD over int16 instead, so nothing saturates and scalar == SIMD
// holds exactly.
//
// Whether the faster path would actually hurt is an empirical question, so it can be measured
// rather than argued: with the probe on, the scalar kernel computes BOTH the exact accumulation and
// a faithful simulation of the u8 path (activations offset to unsigned by a zero point of 128,
// pairwise saturating int16 adds, then the zero-point correction), counts the saturations, and — if
// `substitute` is set — returns the SIMULATED value so the whole model runs on it and the parse can
// be compared name by name.
//
// Scalar path only, and therefore only meaningful with NEURELEASE_SEGMENTER_KERNEL=scalar.
struct UnsignedProbe {
    long long dotProducts = 0;
    long long pairs = 0;
    long long saturatedPairs = 0;      // pair sums that hit the int16 rail
    long long dotProductsAffected = 0; // dot products where the simulated result differs
    long long absoluteErrorSum = 0;    // summed |simulated - exact|, in accumulator units
    int worstAbsoluteError = 0;
};

// --- WHAT THE STRIDE PADDING COSTS ---------------------------------------------------------
//
// Every packed row is rounded up to a multiple of 64 elements so the kernel can load whole vectors
// without a tail. The padding is zeros, so it is correct — but it is also multiplied, and a matrix
// whose real width is 68 pays for 128. These counters say how much of the arithmetic is padding,
// summed over every matrix the model loads, so the question is answerable without reading the
// exporter.
struct PackingStats {
    long long matrices = 0;
    long long widthElements = 0;    // what the model actually needs
    long long strideElements = 0;   // what the kernel walks
    long long strideElements32 = 0;  // what a 32-byte quantum would walk (the avx2 vector width)
    long long strideElements16 = 0;  // what a 16-byte quantum would walk (sse2)
    int worstWidth = 0;             // the matrix that wastes the largest share
    int worstStride = 0;
    // Pair-constraint accounting (zero unless setPairConstraint(true) was on during pack):
    long long pairsChecked = 0;     // same-sign adjacent pairs examined
    long long pairsConstrained = 0; // pairs that exceeded |w0|+|w1| <= 128 and were cut
    long long magnitudeCut = 0;     // total |code| removed across all cut pairs
};
[[nodiscard]] PackingStats packingStats() noexcept;
void resetPackingStats() noexcept;

// `substitute` makes the model USE the simulated numbers; false only counts.
void setUnsignedProbe(bool enabled, bool substitute) noexcept;
[[nodiscard]] UnsignedProbe unsignedProbe() noexcept;
void resetUnsignedProbe() noexcept;

// Every SIMD path must reproduce the scalar path bit for bit, and that is CHECKED, not assumed:
// the first resolve runs the dispatched kernels against scalar on synthetic data covering the
// code extremes. A path that disagrees is demoted to the always-safe scalar fallback for the
// rest of the process, and the demotion is visible here — Segmenter surfaces it per prediction.
struct PathStatus {
    Path active = Path::Scalar;
    bool demoted = false;   // a self-check failed; running on scalar instead of `refused`
    Path refused = Path::Scalar;
};
PathStatus pathStatus() noexcept;

// Optional persistence, call BEFORE first use with a writable file path. Two protections:
// a path that fails the self-check is recorded and never selected again — and a "trying" marker
// is flushed to the file before a SIMD path's first-ever instruction executes and cleared after
// the check returns, so a process that CRASHES in between leaves the marker behind and every
// later run avoids that path without executing it. Without a file, a demotion protects only the
// current process and a crash protects nothing.
void rememberBadPaths(const char *file);

// HOW WIDE A ROW HAS TO BE ROUNDED, in elements, so the widest kernel this CPU can run reads only
// whole vectors and needs no tail loop: 64 where avx512 exists, 32 for the avx2 tiles, 16 for scalar.
//
// It is keyed to what the CPU CAN run, not to what is currently selected, because forcePath() may
// switch to a wider kernel after the matrices are already packed (the cross-path equality test does
// exactly that) and a wider kernel would then read past the row. Padding is zeros either way, so the
// quantum never changes a result — only how many multiplies are spent on nothing.
// NEURELEASE_STRIDE_QUANTUM raises the quantum (never lowers it), so a run can be compared
// against the old flat 64 and the cost of the padding read off directly.
[[nodiscard]] int strideQuantum() noexcept;
[[nodiscard]] int strideQuantum(Path path) noexcept;

// One quantised weight matrix, prepared once at load time for the inner loop:
//   * rows padded to strideQuantum() so every runnable path reads full vectors with no tail loop,
//   * one float scale per row (per-output-channel symmetric, exactly as exported),
//   * per-row code sums, which the u8×s8 paths need for the +128 offset correction.
struct PackedMatrix {
    int rows = 0;
    int width = 0;
    int stride = 0;
    std::vector<std::int8_t> codes;   // rows * stride, padding zero
    std::vector<float> scales;        // rows
    std::vector<std::int32_t> sums;   // rows: sum of the row's codes

    void pack(const std::int8_t *source, const float *rowScales, int rowCount, int rowWidth);
};

// How the active path wants activations stored. The CODES are the same either way — ±127 — only
// the element width differs, so the choice never changes a result.
enum class ActivationKind : std::uint8_t { Int8, Int16 };
ActivationKind activationKind(Path path) noexcept;

// EXACT, OR FAST AND SLIGHTLY WRONG.
//
// Exact is the default and the only mode whose results are bit-identical to the scalar reference on
// every path. Fast swaps the AVX2 matmul for a `vpmaddubsw` one: about half the instructions and half
// the activation bandwidth for the same 32 MACs, at the price of a saturating int16 pair sum. On real
// names that changes the parse of roughly 1% of them. Nothing else is affected - Fast is a no-op on
// scalar, on VNNI, and on AVX-512, whose instructions accumulate into int32 and cannot saturate.
//
// NEURELEASE_SEGMENTER_ACCURACY=fast selects it at startup. Call setAccuracy() only when no parse
// is running: it re-points the dispatch pointers and changes which activation width is expected.
// THREE MODES, and the middle one is the interesting one.
//
//   Exact           the default. int16 activations, `vpmaddwd`, bit-identical to scalar on every path.
//
//   Fast            `vpmaddubsw`: half the instructions and half the activation bandwidth for the same
//                   32 MACs, at the price of a saturating int16 pair sum. How much saturation is a
//                   DIAL, not a fact - see setFastZeroPoint.
//
// Fast is a no-op on scalar, VNNI and AVX-512, whose instructions accumulate straight into int32 and
// cannot saturate - there they are already both faster and exact.
//
// NEURELEASE_SEGMENTER_ACCURACY=fast selects it at startup. Call setAccuracy() only when no parse
// is running: it re-points the dispatch pointers, and it changes both the activation width and the
// range they are quantised to.
enum class Accuracy : std::uint8_t { Exact, Fast };
void setAccuracy(Accuracy accuracy) noexcept;
[[nodiscard]] Accuracy accuracy() noexcept;

// THE ZERO POINT IS THE WHOLE TRADE, and it runs in one dimension.
//
// Activations are clamped to +/-(Z-1) and shifted by Z, so they arrive unsigned in [1, 2Z-1].
// `vpmaddubsw` sums adjacent PAIRS into a signed 16-bit lane, so the worst pair is
// (2Z-1) * 127 * 2 against a ceiling of 32,767: saturation is impossible up to Z = 65 and gets
// progressively more likely above it. But a smaller Z also means a smaller clamp, and therefore
// coarser activations for every value, not just the rare overflowing one.
//
// Both ends hurt and the middle is an empirical question, which is why this is a runtime dial with a
// benchmark behind it (`comparesSpeedOptions` sweeps it). Supported: 64, 80, 96, 112, 128 - 128 being
// full +/-127 activations, which is what FBGEMM does. Speed is identical at every setting; only the
// answers move. NEURELEASE_SEGMENTER_ZERO_POINT overrides the default.
void setFastZeroPoint(int zeroPoint) noexcept;
[[nodiscard]] int fastZeroPoint() noexcept;


// PAIR-CONSTRAINED WEIGHT PACKING: make the fast kernel exact by fixing the WEIGHTS, not the
// activations.
//
// `vpmaddubsw` saturates per adjacent PAIR: u0*w0 + u1*w1 into signed int16. With full-range
// activations (u <= 255) the sum is bounded by 255 * (|w0| + |w1|) when the pair shares a sign, and
// opposite-sign pairs cannot saturate at all. So the exact condition for saturation-freedom is
//
//     |w0| + |w1| <= 128   for every same-sign adjacent weight pair,
//
// because 255 * 128 = 32,640 < 32,767. NOT |w| <= 64 for every weight - a (+127, +1) pair is fine,
// and so is (+127, -127). When this flag is on, pack() walks each row's adjacent pairs in exactly the
// order the kernel consumes them and shrinks only the offending same-sign pairs, by the Euclidean
// projection onto |w0| + |w1| <= 128 (equal cut from both magnitudes). Everything else keeps full
// int8 precision, and the row sums are computed AFTER the cut so the zero-point correction stays
// exact.
//
// The result: the fast u8 kernel becomes bit-identical to the exact kernel ON THESE WEIGHTS, and the
// only remaining error is the (input-independent) weight change itself - measured, not assumed, by
// sweepsTheFastZeroPoint. Set BEFORE the model loads; already-packed matrices are unaffected.
// NEURELEASE_PAIR_CONSTRAINT=1 sets it at startup.
void setPairConstraint(bool enabled) noexcept;
[[nodiscard]] bool pairConstraint() noexcept;

// The largest magnitude an activation code may take under the current mode: 127 normally, and the zero
// point minus one on the u8 path. `quantise` applies it; callers do not need to.
[[nodiscard]] int activationLimit() noexcept;

// What the DISPATCH wants right now - the active path and the accuracy mode together. Prefer this to
// activationKind(activePath()), which cannot see the accuracy mode.
[[nodiscard]] ActivationKind activationKind() noexcept;

// Quantise `count` floats to symmetric codes (round to nearest even, clamp to ±127) and zero the
// buffer through `paddedCount` ELEMENTS, which must be at least the stride of every matrix this
// activation will meet. Returns the scale (max |value| / 127, floored away from zero).
float quantise(const float *values, int count, std::int8_t *destination, int paddedCount) noexcept;
float quantise16(const float *values, int count, std::int16_t *destination,
                 int paddedCount) noexcept;

// accumulators[r] = dot(codes[r], activation) as exact int32, for every row. The activation must
// hold `stride` elements of the active path's ActivationKind with the padding zeroed (see
// quantise / quantise16). Dispatched per CPU.
void matvec(const PackedMatrix &matrix, const void *activation,
            std::int32_t *accumulators) noexcept;

// The batched form, and the one the forward pass actually uses: `actCount` activation rows,
// `actStride` ELEMENTS apart, each read for `stride` elements. accumulators[a * rows + r].
//
// Batching matters twice over. It loads each weight row once per BLOCK of activations instead of
// once per activation — and it is what lets a dilation-1 convolution run with NO im2col at all:
// its windows are overlapping views into the padded map, so actStride is simply the channel
// count, smaller than the window. Rows may overlap arbitrarily; the kernel only reads.
void matmul(const PackedMatrix &matrix, const void *activations, int actCount, int actStride,
            std::int32_t *accumulators) noexcept;

// The shared epilogue: out[r] = bias[r] + accumulators[r] * activationScale * scales[r].
// bias may be nullptr. Kept out of the kernels so every path rounds identically.
void finish(const PackedMatrix &matrix, const std::int32_t *accumulators, float activationScale,
            const float *bias, float *out) noexcept;

// Implementations, one per translation unit. Each returns nullptr when compiled for a target that
// cannot provide it, so the dispatcher never references an instruction the binary lacks. The
// activation pointer's element width is the path's ActivationKind.
using MatvecFunction = void (*)(const PackedMatrix &, const void *, std::int32_t *);
using MatmulFunction = void (*)(const PackedMatrix &, const void *, int, int, std::int32_t *);

// THE EXACT KERNEL, ON DEMAND, whatever the accuracy mode says.
//
// Not every matrix deserves the same trade. The trunk's projections and feed-forward layers are where
// the time is, and an error there is smoothed by everything downstream. The decision HEADS are the
// opposite: they are a few per cent of the arithmetic, and an error in one goes straight into an argmax
// and out as a different answer. So a caller can run the heads exactly and the trunk fast.
//
// `exactMatmul` and `exactActivationKind` describe the exact kernel for the ACTIVE PATH, ignoring the
// accuracy mode; matmul() and activationKind() continue to describe whatever the mode selected.
[[nodiscard]] MatmulFunction exactMatmul() noexcept;
[[nodiscard]] ActivationKind exactActivationKind() noexcept;
MatvecFunction scalarMatvec() noexcept;
MatvecFunction avx2Matvec() noexcept;       // SegmenterKernelsAvx2.cpp,   -mavx2
MatvecFunction avx2VnniMatvec() noexcept;   // SegmenterKernelsAvxVnni.cpp, -mavxvnni
MatvecFunction avx512VnniMatvec() noexcept; // SegmenterKernelsAvx512.cpp, -mavx512vnni
MatmulFunction avx2Matmul() noexcept;       // row-blocked register tiles, one per VNNI width
// SegmenterKernelsAvx2Fast.cpp, -mavx2. One instantiation per supported zero point; nullptr for an
// unsupported one, and nullptr throughout where the build has no AVX2.
MatvecFunction avx2FastMatvec(int zeroPoint) noexcept;
MatmulFunction avx2FastMatmul(int zeroPoint) noexcept;
MatmulFunction avx2VnniMatmul() noexcept;
MatmulFunction avx512VnniMatmul() noexcept;

} // namespace neurelease::model::kernels
