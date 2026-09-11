// AVX2 int8-weight matvec — the path every x86-64 CPU since Haswell (2013) and Zen 1 (2017) can
// run, and the BEST path on all consumer AMD parts before Zen 4, which have no VNNI.
//
// Deliberately NOT `vpmaddubsw`: that instruction is unsigned×signed and saturates its int16
// intermediate (255·127 + 255·127 = 64,770 against a ceiling of 32,767 — real activations hit
// it). `vpmaddwd` over int16 operands is signed, exact, and needs no +128 offset bookkeeping.
//
// The activation arrives ALREADY widened to int16 (ActivationKind::Int16): widening it once per
// matrix instead of once per weight row removes two register conversions per 32 weights, which
// halves the inner loop's instruction count. Weights are still widened in the loop — they change
// every row, so nothing is saved by pre-widening them, and int8 storage halves their cache
// footprint. Two weight rows per pass share each activation load.
//
// Compiled with -mavx2 (or /arch:AVX2) for THIS file only; the dispatcher guarantees it is never
// called on a CPU without AVX2.

#include "model/kernels/kernels.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(_MSC_VER))
#define NEURELEASE_HAVE_AVX2 1
#include <immintrin.h>
#endif

namespace neurelease::model::kernels {

#ifdef NEURELEASE_HAVE_AVX2

namespace {

inline std::int32_t horizontalSum(__m256i vector) noexcept {
    const __m128i low = _mm256_castsi256_si128(vector);
    const __m128i high = _mm256_extracti128_si256(vector, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

void matvecAvx2(const PackedMatrix &matrix, const void *activation,
                std::int32_t *accumulators) {
    const auto *wide = static_cast<const std::int16_t *>(activation);
    const int stride = matrix.stride;
    int row = 0;
    for (; row + 1 < matrix.rows; row += 2) {
        const std::int8_t *first = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int8_t *second = first + stride;
        __m256i sumFirstA = _mm256_setzero_si256(), sumFirstB = _mm256_setzero_si256();
        __m256i sumSecondA = _mm256_setzero_si256(), sumSecondB = _mm256_setzero_si256();
        for (int at = 0; at < stride; at += 32) {
            const __m256i actLow =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wide + at));
            const __m256i actHigh =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wide + at + 16));

            const __m256i firstBytes =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at));
            sumFirstA = _mm256_add_epi32(
                sumFirstA, _mm256_madd_epi16(
                               actLow, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(firstBytes))));
            sumFirstB = _mm256_add_epi32(
                sumFirstB,
                _mm256_madd_epi16(actHigh,
                                  _mm256_cvtepi8_epi16(_mm256_extracti128_si256(firstBytes, 1))));

            const __m256i secondBytes =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at));
            sumSecondA = _mm256_add_epi32(
                sumSecondA,
                _mm256_madd_epi16(actLow,
                                  _mm256_cvtepi8_epi16(_mm256_castsi256_si128(secondBytes))));
            sumSecondB = _mm256_add_epi32(
                sumSecondB,
                _mm256_madd_epi16(actHigh,
                                  _mm256_cvtepi8_epi16(_mm256_extracti128_si256(secondBytes, 1))));
        }
        accumulators[row] = horizontalSum(_mm256_add_epi32(sumFirstA, sumFirstB));
        accumulators[row + 1] = horizontalSum(_mm256_add_epi32(sumSecondA, sumSecondB));
    }
    for (; row < matrix.rows; ++row) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        __m256i sumA = _mm256_setzero_si256(), sumB = _mm256_setzero_si256();
        for (int at = 0; at < stride; at += 32) {
            const __m256i weightBytes =
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at));
            sumA = _mm256_add_epi32(
                sumA,
                _mm256_madd_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wide + at)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(weightBytes))));
            sumB = _mm256_add_epi32(
                sumB,
                _mm256_madd_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(wide + at + 16)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weightBytes, 1))));
        }
        accumulators[row] = horizontalSum(_mm256_add_epi32(sumA, sumB));
    }
}

// The batched kernel, blocked the way FBGEMM and oneDNN block their int8 GEMMs: a register tile
// of 2 weight rows × 4 activation rows, so each weight chunk is loaded and widened ONCE per four
// activations, each activation chunk loaded once per two weight rows, and all eight accumulators
// live in registers across the whole reduction. This is where batch mode earns its keep — the
// matvec above re-reads the entire weight matrix for every activation row.
//
// `actStride` may be SMALLER than the width being read: a dilation-1 convolution hands the map
// itself as overlapping windows, actStride = channels, and skips im2col entirely.
//
// Accumulation is int32 and exact (max |sum| ≈ width · 127² ≪ 2³¹), so any summation order gives
// the same answer — the bit-exactness contract with the scalar path survives the tiling.
void matmulAvx2(const PackedMatrix &matrix, const void *activations, int actCount, int actStride,
                std::int32_t *accumulators) {
    const auto *base = static_cast<const std::int16_t *>(activations);
    const int stride = matrix.stride;
    const int rows = matrix.rows;

    int row = 0;
    for (; row + 1 < rows; row += 2) {
        const std::int8_t *first = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int8_t *second = first + stride;

        int act = 0;
        for (; act + 3 < actCount; act += 4) {
            const std::int16_t *a0 = base + static_cast<std::size_t>(act) * actStride;
            const std::int16_t *a1 = a0 + actStride;
            const std::int16_t *a2 = a1 + actStride;
            const std::int16_t *a3 = a2 + actStride;
            __m256i acc00 = _mm256_setzero_si256(), acc01 = _mm256_setzero_si256();
            __m256i acc02 = _mm256_setzero_si256(), acc03 = _mm256_setzero_si256();
            __m256i acc10 = _mm256_setzero_si256(), acc11 = _mm256_setzero_si256();
            __m256i acc12 = _mm256_setzero_si256(), acc13 = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                const __m256i firstBytes =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at));
                const __m256i w0lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(firstBytes));
                const __m256i w0hi =
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(firstBytes, 1));
                const __m256i secondBytes =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at));
                const __m256i w1lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(secondBytes));
                const __m256i w1hi =
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(secondBytes, 1));

                __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at));
                __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at + 16));
                acc00 = _mm256_add_epi32(acc00, _mm256_madd_epi16(lo, w0lo));
                acc00 = _mm256_add_epi32(acc00, _mm256_madd_epi16(hi, w0hi));
                acc10 = _mm256_add_epi32(acc10, _mm256_madd_epi16(lo, w1lo));
                acc10 = _mm256_add_epi32(acc10, _mm256_madd_epi16(hi, w1hi));

                lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a1 + at));
                hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a1 + at + 16));
                acc01 = _mm256_add_epi32(acc01, _mm256_madd_epi16(lo, w0lo));
                acc01 = _mm256_add_epi32(acc01, _mm256_madd_epi16(hi, w0hi));
                acc11 = _mm256_add_epi32(acc11, _mm256_madd_epi16(lo, w1lo));
                acc11 = _mm256_add_epi32(acc11, _mm256_madd_epi16(hi, w1hi));

                lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a2 + at));
                hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a2 + at + 16));
                acc02 = _mm256_add_epi32(acc02, _mm256_madd_epi16(lo, w0lo));
                acc02 = _mm256_add_epi32(acc02, _mm256_madd_epi16(hi, w0hi));
                acc12 = _mm256_add_epi32(acc12, _mm256_madd_epi16(lo, w1lo));
                acc12 = _mm256_add_epi32(acc12, _mm256_madd_epi16(hi, w1hi));

                lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a3 + at));
                hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a3 + at + 16));
                acc03 = _mm256_add_epi32(acc03, _mm256_madd_epi16(lo, w0lo));
                acc03 = _mm256_add_epi32(acc03, _mm256_madd_epi16(hi, w0hi));
                acc13 = _mm256_add_epi32(acc13, _mm256_madd_epi16(lo, w1lo));
                acc13 = _mm256_add_epi32(acc13, _mm256_madd_epi16(hi, w1hi));
            }
            std::int32_t *out0 = accumulators + static_cast<std::size_t>(act) * rows + row;
            out0[0] = horizontalSum(acc00);
            out0[1] = horizontalSum(acc10);
            std::int32_t *out1 = out0 + rows;
            out1[0] = horizontalSum(acc01);
            out1[1] = horizontalSum(acc11);
            std::int32_t *out2 = out1 + rows;
            out2[0] = horizontalSum(acc02);
            out2[1] = horizontalSum(acc12);
            std::int32_t *out3 = out2 + rows;
            out3[0] = horizontalSum(acc03);
            out3[1] = horizontalSum(acc13);
        }
        for (; act < actCount; ++act) {
            const std::int16_t *a0 = base + static_cast<std::size_t>(act) * actStride;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                const __m256i lo =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at));
                const __m256i hi =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at + 16));
                const __m256i firstBytes =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at));
                acc0 = _mm256_add_epi32(
                    acc0, _mm256_madd_epi16(
                              lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(firstBytes))));
                acc0 = _mm256_add_epi32(
                    acc0, _mm256_madd_epi16(
                              hi, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(firstBytes, 1))));
                const __m256i secondBytes =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at));
                acc1 = _mm256_add_epi32(
                    acc1, _mm256_madd_epi16(
                              lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(secondBytes))));
                acc1 = _mm256_add_epi32(
                    acc1, _mm256_madd_epi16(
                              hi, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(secondBytes, 1))));
            }
            std::int32_t *out = accumulators + static_cast<std::size_t>(act) * rows + row;
            out[0] = horizontalSum(acc0);
            out[1] = horizontalSum(acc1);
        }
    }
    if (row < rows) { // odd final weight row
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        for (int act = 0; act < actCount; ++act) {
            const std::int16_t *a0 = base + static_cast<std::size_t>(act) * actStride;
            __m256i acc = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                const __m256i weightBytes =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at));
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at)),
                             _mm256_cvtepi8_epi16(_mm256_castsi256_si128(weightBytes))));
                acc = _mm256_add_epi32(
                    acc,
                    _mm256_madd_epi16(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at + 16)),
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weightBytes, 1))));
            }
            accumulators[static_cast<std::size_t>(act) * rows + row] = horizontalSum(acc);
        }
    }
}

} // namespace

MatvecFunction avx2Matvec() noexcept { return &matvecAvx2; }

MatmulFunction avx2Matmul() noexcept { return &matmulAvx2; }

#else

MatvecFunction avx2Matvec() noexcept { return nullptr; }

MatmulFunction avx2Matmul() noexcept { return nullptr; }

#endif

} // namespace neurelease::model::kernels
