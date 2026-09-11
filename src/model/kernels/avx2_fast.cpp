// AVX2 int8 kernels that trade exactness for throughput — `vpmaddubsw` instead of `vpmaddwd`.
//
// The exact AVX2 path in SegmenterKernelsAvx2.cpp multiplies int16 activations by sign-extended
// int16 weights: per 32 weights it costs two activation loads, one weight load, two sign extensions,
// two `vpmaddwd` and two adds. This path does the same 32 MACs with one u8 activation load, one
// weight load, one `vpmaddubsw`, one `vpmaddwd` against ones, and one add — roughly half the
// instructions and half the activation bandwidth.
//
// THE SATURATION, AND HOW TO MAKE IT IMPOSSIBLE. `vpmaddubsw` is unsigned × signed and adds each
// adjacent PAIR into a signed 16-bit lane, so at the full range it overflows: 255*127 + 255*127 =
// 64,770 against a ceiling of 32,767. Measured on 2,000 real names, 0.0135% of pairs saturate, which
// poisons 2.54% of dot products (one bad pair spoils its whole dot product) and changes the final
// parse of about 1.1% of names.
//
// Lowering the ZERO POINT bounds it: at Z, activations clamp to +/-(Z-1) and arrive in [1, 2Z-1], so
// the worst pair is (2Z-1) * 127 * 2 and saturation is arithmetically impossible up to Z = 65.
//
// AND IT IS NOT WORTH GOING THERE. Measured over 2,000 real names, Z = 64 changes 1.80% of parses
// where Z = 128 changes 1.10%: the activation resolution given up to buy safety costs more than the
// overflow it prevents, because the clamp is paid by every value and the saturation by very few. The
// template covers 64 through 128 so the whole curve can be swept rather than argued about; the default
// is 128.
//
// The bias bookkeeping matches the VNNI path: activations are shifted to unsigned and
// ZeroPoint * sum(row codes) is subtracted afterwards, using the per-row sums PackedMatrix
// precomputed at load. That part is exact at either zero point.
//
// Compiled with -mavx2 for THIS file only.

#include "model/kernels/kernels.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(_MSC_VER))
#define NEURELEASE_HAVE_AVX2_FAST 1
#include <immintrin.h>
#endif

#include <vector>

namespace neurelease::model::kernels {

#ifdef NEURELEASE_HAVE_AVX2_FAST

namespace {

constexpr int WidestActivation = 4096;

inline std::int32_t horizontalSum(__m256i vector) noexcept {
    const __m128i low = _mm256_castsi256_si128(vector);
    const __m128i high = _mm256_extracti128_si256(vector, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

// One fused multiply-accumulate step: 32 u8 activations against 32 int8 weights, widened to int32.
inline __m256i step(__m256i accumulator, __m256i activation, __m256i weights,
                    __m256i ones) noexcept {
    return _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(_mm256_maddubs_epi16(activation, weights), ones));
}

// Signed activation codes to unsigned, by the zero point. At 128 the shift is an XOR of the sign bit;
// anywhere else it is a real add, which wraps modulo 256 exactly as wanted - the sum can exceed int8
// range and the BIT PATTERN is still the right unsigned value, which is all `vpmaddubsw` reads. Safe
// because the codes were clamped to the zero point minus one on the way in (kernels::activationLimit).
template <int ZeroPoint>
inline __m256i bias(__m256i codes) noexcept {
    if constexpr (ZeroPoint == 128) {
        return _mm256_xor_si256(codes, _mm256_set1_epi8(static_cast<char>(0x80)));
    } else {
        return _mm256_add_epi8(codes, _mm256_set1_epi8(static_cast<char>(ZeroPoint)));
    }
}

template <int ZeroPoint>
void matvecAvx2Fast(const PackedMatrix &matrix, const void *rawActivation,
                    std::int32_t *accumulators) {
    const auto *activation = static_cast<const std::int8_t *>(rawActivation);
    if (matrix.stride > WidestActivation) {
        scalarMatvec()(matrix, activation, accumulators);
        return;
    }
    alignas(32) std::uint8_t biased[WidestActivation];
    const int stride = matrix.stride;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int at = 0; at < stride; at += 32) {
        _mm256_store_si256(reinterpret_cast<__m256i *>(biased + at),
                           bias<ZeroPoint>(_mm256_loadu_si256(
                               reinterpret_cast<const __m256i *>(activation + at))));
    }

    int row = 0;
    for (; row + 1 < matrix.rows; row += 2) {
        const std::int8_t *first = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int8_t *second = first + stride;
        __m256i sumFirst = _mm256_setzero_si256();
        __m256i sumSecond = _mm256_setzero_si256();
        for (int at = 0; at < stride; at += 32) {
            const __m256i value =
                _mm256_load_si256(reinterpret_cast<const __m256i *>(biased + at));
            sumFirst = step(sumFirst, value,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at)),
                            ones);
            sumSecond = step(sumSecond, value,
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at)),
                             ones);
        }
        accumulators[row] =
            horizontalSum(sumFirst) - ZeroPoint * matrix.sums[static_cast<std::size_t>(row)];
        accumulators[row + 1] =
            horizontalSum(sumSecond) - ZeroPoint * matrix.sums[static_cast<std::size_t>(row) + 1];
    }
    if (row < matrix.rows) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        __m256i sum = _mm256_setzero_si256();
        for (int at = 0; at < stride; at += 32) {
            sum = step(sum, _mm256_load_si256(reinterpret_cast<const __m256i *>(biased + at)),
                       _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at)), ones);
        }
        accumulators[row] =
            horizontalSum(sum) - ZeroPoint * matrix.sums[static_cast<std::size_t>(row)];
    }
}

// One biased copy of the activation region per call, reused by every tile. Thread-local so
// concurrent parses never share it. Deliberately never destroyed — see SegmenterModel.cpp: this
// toolchain corrupts the heap freeing thread_local storage at thread exit.
thread_local std::vector<std::uint8_t> &biasedStorage = *new std::vector<std::uint8_t>;

template <int ZeroPoint>
void matmulAvx2Fast(const PackedMatrix &matrix, const void *activations, int actCount,
                    int actStride, std::int32_t *accumulators) {
    const auto *base = static_cast<const std::int8_t *>(activations);
    const int stride = matrix.stride;
    const int rows = matrix.rows;
    const __m256i ones = _mm256_set1_epi16(1);

    // Callers' buffers all carry at least a 64-element tail past the region, so rounding the
    // conversion up to a whole vector stays in bounds.
    const std::size_t region = static_cast<std::size_t>(actCount - 1) * actStride + stride;
    auto &biased = biasedStorage;
    biased.resize((region + 31) / 32 * 32);
    for (std::size_t at = 0; at < region; at += 32) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(biased.data() + at),
                            bias<ZeroPoint>(_mm256_loadu_si256(
                                reinterpret_cast<const __m256i *>(base + at))));
    }

    int row = 0;
    for (; row + 1 < rows; row += 2) {
        const std::int8_t *first = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int8_t *second = first + stride;
        const std::int32_t correctFirst = ZeroPoint * matrix.sums[static_cast<std::size_t>(row)];
        const std::int32_t correctSecond =
            ZeroPoint * matrix.sums[static_cast<std::size_t>(row) + 1];

        int act = 0;
        for (; act + 3 < actCount; act += 4) {
            const std::uint8_t *a0 = biased.data() + static_cast<std::size_t>(act) * actStride;
            const std::uint8_t *a1 = a0 + actStride;
            const std::uint8_t *a2 = a1 + actStride;
            const std::uint8_t *a3 = a2 + actStride;
            __m256i acc00 = _mm256_setzero_si256(), acc01 = _mm256_setzero_si256();
            __m256i acc02 = _mm256_setzero_si256(), acc03 = _mm256_setzero_si256();
            __m256i acc10 = _mm256_setzero_si256(), acc11 = _mm256_setzero_si256();
            __m256i acc12 = _mm256_setzero_si256(), acc13 = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                const __m256i w0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at));
                const __m256i w1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at));
                const __m256i v0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at));
                acc00 = step(acc00, v0, w0, ones);
                acc10 = step(acc10, v0, w1, ones);
                const __m256i v1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a1 + at));
                acc01 = step(acc01, v1, w0, ones);
                acc11 = step(acc11, v1, w1, ones);
                const __m256i v2 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a2 + at));
                acc02 = step(acc02, v2, w0, ones);
                acc12 = step(acc12, v2, w1, ones);
                const __m256i v3 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a3 + at));
                acc03 = step(acc03, v3, w0, ones);
                acc13 = step(acc13, v3, w1, ones);
            }
            std::int32_t *out0 = accumulators + static_cast<std::size_t>(act) * rows + row;
            out0[0] = horizontalSum(acc00) - correctFirst;
            out0[1] = horizontalSum(acc10) - correctSecond;
            std::int32_t *out1 = out0 + rows;
            out1[0] = horizontalSum(acc01) - correctFirst;
            out1[1] = horizontalSum(acc11) - correctSecond;
            std::int32_t *out2 = out1 + rows;
            out2[0] = horizontalSum(acc02) - correctFirst;
            out2[1] = horizontalSum(acc12) - correctSecond;
            std::int32_t *out3 = out2 + rows;
            out3[0] = horizontalSum(acc03) - correctFirst;
            out3[1] = horizontalSum(acc13) - correctSecond;
        }
        for (; act < actCount; ++act) {
            const std::uint8_t *a0 = biased.data() + static_cast<std::size_t>(act) * actStride;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                const __m256i v0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at));
                acc0 = step(acc0, v0,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at)),
                            ones);
                acc1 = step(acc1, v0,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at)),
                            ones);
            }
            std::int32_t *out = accumulators + static_cast<std::size_t>(act) * rows + row;
            out[0] = horizontalSum(acc0) - correctFirst;
            out[1] = horizontalSum(acc1) - correctSecond;
        }
    }
    if (row < rows) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int32_t correct = ZeroPoint * matrix.sums[static_cast<std::size_t>(row)];
        for (int act = 0; act < actCount; ++act) {
            const std::uint8_t *a0 = biased.data() + static_cast<std::size_t>(act) * actStride;
            __m256i acc = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                acc = step(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at)),
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at)),
                           ones);
            }
            accumulators[static_cast<std::size_t>(act) * rows + row] =
                horizontalSum(acc) - correct;
        }
    }
}

} // namespace

MatvecFunction avx2FastMatvec(int zeroPoint) noexcept {
    switch (zeroPoint) {
    case 64:
        return &matvecAvx2Fast<64>;
    case 80:
        return &matvecAvx2Fast<80>;
    case 96:
        return &matvecAvx2Fast<96>;
    case 112:
        return &matvecAvx2Fast<112>;
    case 128:
        return &matvecAvx2Fast<128>;
    }
    return nullptr;
}

MatmulFunction avx2FastMatmul(int zeroPoint) noexcept {
    switch (zeroPoint) {
    case 64:
        return &matmulAvx2Fast<64>;
    case 80:
        return &matmulAvx2Fast<80>;
    case 96:
        return &matmulAvx2Fast<96>;
    case 112:
        return &matmulAvx2Fast<112>;
    case 128:
        return &matmulAvx2Fast<128>;
    }
    return nullptr;
}

#else

MatvecFunction avx2FastMatvec(int) noexcept { return nullptr; }

MatmulFunction avx2FastMatmul(int) noexcept { return nullptr; }

#endif

} // namespace neurelease::model::kernels
