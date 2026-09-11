// AVX2-VNNI int8 kernels — 256-bit `vpdpbusd` on CPUs that have VNNI without AVX-512 (Intel
// Alder/Raptor Lake E-core designs, and Zen 5's 256-bit datapath).
//
// `vpdpbusd` is unsigned×signed: activations are biased to unsigned with an XOR 0x80 and the bias
// is removed afterwards as 128 * sum(row codes), which PackedMatrix precomputed at load. The
// instruction accumulates directly into int32, so unlike `vpmaddubsw` there is no int16
// saturation to defend against — the result is exact and bit-identical to the scalar path.
//
// The batched matmul biases the whole activation region ONCE per call and then runs the same
// 2 weight row × 4 activation row register tile as the AVX2 kernel, so each weight chunk is
// loaded once per four activations. Rows may overlap (a dilation-1 convolution passes its map as
// overlapping windows), which is why the region is converted as one span rather than row by row.
//
// Compiled with -mavxvnni for THIS file only. On MSVC the intrinsic needs VS2022 17.2+; older
// compilers simply build the nullptr stubs and the dispatcher never offers the path.

#include "model/kernels/kernels.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__AVXVNNI__) || (defined(_MSC_VER) && _MSC_VER >= 1932))
#define NEURELEASE_HAVE_AVXVNNI 1
#include <immintrin.h>
#endif

namespace neurelease::model::kernels {

#ifdef NEURELEASE_HAVE_AVXVNNI

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

void matvecAvxVnni(const PackedMatrix &matrix, const void *rawActivation,
                   std::int32_t *accumulators) {
    const auto *activation = static_cast<const std::int8_t *>(rawActivation);
    if (matrix.stride > WidestActivation) {
        scalarMatvec()(matrix, activation, accumulators);
        return;
    }
    alignas(32) std::uint8_t biased[WidestActivation];
    const int stride = matrix.stride;
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    for (int at = 0; at < stride; at += 32) {
        _mm256_store_si256(
            reinterpret_cast<__m256i *>(biased + at),
            _mm256_xor_si256(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(activation + at)), flip));
    }
    for (int row = 0; row < matrix.rows; ++row) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        __m256i sumA = _mm256_setzero_si256();
        __m256i sumB = _mm256_setzero_si256();
        int at = 0;
        for (; at + 64 <= stride; at += 64) {
            sumA = _mm256_dpbusd_avx_epi32(
                sumA, _mm256_load_si256(reinterpret_cast<const __m256i *>(biased + at)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at)));
            sumB = _mm256_dpbusd_avx_epi32(
                sumB, _mm256_load_si256(reinterpret_cast<const __m256i *>(biased + at + 32)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at + 32)));
        }
        for (; at < stride; at += 32) {
            sumA = _mm256_dpbusd_avx_epi32(
                sumA, _mm256_load_si256(reinterpret_cast<const __m256i *>(biased + at)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at)));
        }
        accumulators[row] = horizontalSum(_mm256_add_epi32(sumA, sumB)) -
                            128 * matrix.sums[static_cast<std::size_t>(row)];
    }
}

// One biased copy of the activation region per matmul call, reused by every tile. Thread-local so
// concurrent parses never share it; it grows to the largest region seen and then stops
// allocating.
thread_local std::vector<std::uint8_t> biasedStorage;

void matmulAvxVnni(const PackedMatrix &matrix, const void *activations, int actCount,
                   int actStride, std::int32_t *accumulators) {
    const auto *base = static_cast<const std::int8_t *>(activations);
    const int stride = matrix.stride;
    const int rows = matrix.rows;

    // The callers' buffers all carry at least a 64-element tail past the region (the packed
    // stride is a multiple of 64, and the convolution map allocates one explicitly), so rounding
    // the conversion up to a whole vector stays in bounds.
    const std::size_t region = static_cast<std::size_t>(actCount - 1) * actStride + stride;
    auto &biased = biasedStorage;
    biased.resize((region + 31) / 32 * 32);
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    for (std::size_t at = 0; at < region; at += 32) {
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>(biased.data() + at),
            _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(base + at)),
                             flip));
    }

    int row = 0;
    for (; row + 1 < rows; row += 2) {
        const std::int8_t *first = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int8_t *second = first + stride;
        const std::int32_t correctFirst = 128 * matrix.sums[static_cast<std::size_t>(row)];
        const std::int32_t correctSecond = 128 * matrix.sums[static_cast<std::size_t>(row) + 1];

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
                acc00 = _mm256_dpbusd_avx_epi32(acc00, v0, w0);
                acc10 = _mm256_dpbusd_avx_epi32(acc10, v0, w1);
                const __m256i v1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a1 + at));
                acc01 = _mm256_dpbusd_avx_epi32(acc01, v1, w0);
                acc11 = _mm256_dpbusd_avx_epi32(acc11, v1, w1);
                const __m256i v2 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a2 + at));
                acc02 = _mm256_dpbusd_avx_epi32(acc02, v2, w0);
                acc12 = _mm256_dpbusd_avx_epi32(acc12, v2, w1);
                const __m256i v3 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a3 + at));
                acc03 = _mm256_dpbusd_avx_epi32(acc03, v3, w0);
                acc13 = _mm256_dpbusd_avx_epi32(acc13, v3, w1);
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
                acc0 = _mm256_dpbusd_avx_epi32(
                    acc0, v0, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(first + at)));
                acc1 = _mm256_dpbusd_avx_epi32(
                    acc1, v0,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(second + at)));
            }
            std::int32_t *out = accumulators + static_cast<std::size_t>(act) * rows + row;
            out[0] = horizontalSum(acc0) - correctFirst;
            out[1] = horizontalSum(acc1) - correctSecond;
        }
    }
    if (row < rows) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int32_t correct = 128 * matrix.sums[static_cast<std::size_t>(row)];
        for (int act = 0; act < actCount; ++act) {
            const std::uint8_t *a0 = biased.data() + static_cast<std::size_t>(act) * actStride;
            __m256i acc = _mm256_setzero_si256();
            for (int at = 0; at < stride; at += 32) {
                acc = _mm256_dpbusd_avx_epi32(
                    acc, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a0 + at)),
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights + at)));
            }
            accumulators[static_cast<std::size_t>(act) * rows + row] =
                horizontalSum(acc) - correct;
        }
    }
}

} // namespace

MatvecFunction avx2VnniMatvec() noexcept { return &matvecAvxVnni; }

MatmulFunction avx2VnniMatmul() noexcept { return &matmulAvxVnni; }

#else

MatvecFunction avx2VnniMatvec() noexcept { return nullptr; }

MatmulFunction avx2VnniMatmul() noexcept { return nullptr; }

#endif

} // namespace neurelease::model::kernels
