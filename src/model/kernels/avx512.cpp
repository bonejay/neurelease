// AVX-512 VNNI int8 kernels — `vpdpbusd` over 64-byte vectors (Ice Lake, Zen 4 and later).
//
// Same unsigned×signed contract and the same +128 bias trick as the 256-bit VNNI path; see
// SegmenterKernelsAvxVnni.cpp. The batched matmul runs the same 2 weight row × 4 activation row
// register tile, in 512-bit vectors: one 64-byte step covers this model's whole token width, so
// on typical rows the inner loop is a handful of dpbusd instructions per output.
//
// Compiled with -mavx512vnni (and f/bw/vl) for THIS file only; the dispatcher additionally checks
// XCR0, because AVX-512 state exists only when the OS saves it.

#include "model/kernels/kernels.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__)))
#define NEURELEASE_HAVE_AVX512VNNI 1
#include <immintrin.h>
#endif

namespace neurelease::model::kernels {

#ifdef NEURELEASE_HAVE_AVX512VNNI

namespace {

constexpr int WidestActivation = 4096;

void matvecAvx512(const PackedMatrix &matrix, const void *rawActivation,
                  std::int32_t *accumulators) {
    const auto *activation = static_cast<const std::int8_t *>(rawActivation);
    if (matrix.stride > WidestActivation) {
        scalarMatvec()(matrix, activation, accumulators);
        return;
    }
    alignas(64) std::uint8_t biased[WidestActivation];
    const int stride = matrix.stride;
    const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
    for (int at = 0; at < stride; at += 64) {
        _mm512_store_si512(biased + at,
                           _mm512_xor_si512(_mm512_loadu_si512(activation + at), flip));
    }
    for (int row = 0; row < matrix.rows; ++row) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        __m512i sums[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                           _mm512_setzero_si512(), _mm512_setzero_si512()};
        int at = 0;
        int lane = 0;
        for (; at < stride; at += 64, lane = (lane + 1) & 3) {
            sums[lane] = _mm512_dpbusd_epi32(sums[lane], _mm512_load_si512(biased + at),
                                             _mm512_loadu_si512(weights + at));
        }
        const __m512i total = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]),
                                               _mm512_add_epi32(sums[2], sums[3]));
        accumulators[row] = _mm512_reduce_add_epi32(total) -
                            128 * matrix.sums[static_cast<std::size_t>(row)];
    }
}

// See SegmenterKernelsAvxVnni.cpp: one biased copy of the whole (possibly overlapping-row)
// activation region per matmul call.
thread_local std::vector<std::uint8_t> biasedStorage;

void matmulAvx512(const PackedMatrix &matrix, const void *activations, int actCount,
                  int actStride, std::int32_t *accumulators) {
    const auto *base = static_cast<const std::int8_t *>(activations);
    const int stride = matrix.stride;
    const int rows = matrix.rows;

    const std::size_t region = static_cast<std::size_t>(actCount - 1) * actStride + stride;
    auto &biased = biasedStorage;
    biased.resize((region + 63) / 64 * 64);
    const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
    for (std::size_t at = 0; at < region; at += 64) {
        _mm512_storeu_si512(biased.data() + at,
                            _mm512_xor_si512(_mm512_loadu_si512(base + at), flip));
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
            __m512i acc00 = _mm512_setzero_si512(), acc01 = _mm512_setzero_si512();
            __m512i acc02 = _mm512_setzero_si512(), acc03 = _mm512_setzero_si512();
            __m512i acc10 = _mm512_setzero_si512(), acc11 = _mm512_setzero_si512();
            __m512i acc12 = _mm512_setzero_si512(), acc13 = _mm512_setzero_si512();
            for (int at = 0; at < stride; at += 64) {
                const __m512i w0 = _mm512_loadu_si512(first + at);
                const __m512i w1 = _mm512_loadu_si512(second + at);
                const __m512i v0 = _mm512_loadu_si512(a0 + at);
                acc00 = _mm512_dpbusd_epi32(acc00, v0, w0);
                acc10 = _mm512_dpbusd_epi32(acc10, v0, w1);
                const __m512i v1 = _mm512_loadu_si512(a1 + at);
                acc01 = _mm512_dpbusd_epi32(acc01, v1, w0);
                acc11 = _mm512_dpbusd_epi32(acc11, v1, w1);
                const __m512i v2 = _mm512_loadu_si512(a2 + at);
                acc02 = _mm512_dpbusd_epi32(acc02, v2, w0);
                acc12 = _mm512_dpbusd_epi32(acc12, v2, w1);
                const __m512i v3 = _mm512_loadu_si512(a3 + at);
                acc03 = _mm512_dpbusd_epi32(acc03, v3, w0);
                acc13 = _mm512_dpbusd_epi32(acc13, v3, w1);
            }
            std::int32_t *out0 = accumulators + static_cast<std::size_t>(act) * rows + row;
            out0[0] = _mm512_reduce_add_epi32(acc00) - correctFirst;
            out0[1] = _mm512_reduce_add_epi32(acc10) - correctSecond;
            std::int32_t *out1 = out0 + rows;
            out1[0] = _mm512_reduce_add_epi32(acc01) - correctFirst;
            out1[1] = _mm512_reduce_add_epi32(acc11) - correctSecond;
            std::int32_t *out2 = out1 + rows;
            out2[0] = _mm512_reduce_add_epi32(acc02) - correctFirst;
            out2[1] = _mm512_reduce_add_epi32(acc12) - correctSecond;
            std::int32_t *out3 = out2 + rows;
            out3[0] = _mm512_reduce_add_epi32(acc03) - correctFirst;
            out3[1] = _mm512_reduce_add_epi32(acc13) - correctSecond;
        }
        for (; act < actCount; ++act) {
            const std::uint8_t *a0 = biased.data() + static_cast<std::size_t>(act) * actStride;
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            for (int at = 0; at < stride; at += 64) {
                const __m512i v0 = _mm512_loadu_si512(a0 + at);
                acc0 = _mm512_dpbusd_epi32(acc0, v0, _mm512_loadu_si512(first + at));
                acc1 = _mm512_dpbusd_epi32(acc1, v0, _mm512_loadu_si512(second + at));
            }
            std::int32_t *out = accumulators + static_cast<std::size_t>(act) * rows + row;
            out[0] = _mm512_reduce_add_epi32(acc0) - correctFirst;
            out[1] = _mm512_reduce_add_epi32(acc1) - correctSecond;
        }
    }
    if (row < rows) {
        const std::int8_t *weights = matrix.codes.data() + static_cast<std::size_t>(row) * stride;
        const std::int32_t correct = 128 * matrix.sums[static_cast<std::size_t>(row)];
        for (int act = 0; act < actCount; ++act) {
            const std::uint8_t *a0 = biased.data() + static_cast<std::size_t>(act) * actStride;
            __m512i acc = _mm512_setzero_si512();
            for (int at = 0; at < stride; at += 64) {
                acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(a0 + at),
                                          _mm512_loadu_si512(weights + at));
            }
            accumulators[static_cast<std::size_t>(act) * rows + row] =
                _mm512_reduce_add_epi32(acc) - correct;
        }
    }
}

} // namespace

MatvecFunction avx512VnniMatvec() noexcept { return &matvecAvx512; }

MatmulFunction avx512VnniMatmul() noexcept { return &matmulAvx512; }

#else

MatvecFunction avx512VnniMatvec() noexcept { return nullptr; }

MatmulFunction avx512VnniMatmul() noexcept { return nullptr; }

#endif

} // namespace neurelease::model::kernels
