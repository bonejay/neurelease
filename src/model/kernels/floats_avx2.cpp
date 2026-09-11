// AVX2 + FMA float helpers for the forward pass: layer norm, GELU, softmax, and the two attention
// primitives. Eight lanes and a fused multiply-add where the baseline had four lanes and separate
// multiply and add.
//
// Compiled with -mavx2 -mfma for THIS file only; the dispatcher never calls it on a CPU that cannot.

#include "model/kernels/floats.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(_MSC_VER))
#define NEURELEASE_HAVE_AVX2_FLOATS 1
#include <immintrin.h>
#endif

#include <cmath>
#include <cstring>
#include <vector>

namespace neurelease::model::floats {

#ifdef NEURELEASE_HAVE_AVX2_FLOATS

namespace {

inline float horizontalSum(__m256 vector) noexcept {
    const __m128 low = _mm256_castps256_ps128(vector);
    const __m128 high = _mm256_extractf128_ps(vector, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x1));
    return _mm_cvtss_f32(sum);
}

// exp by range reduction and a degree-6 Taylor polynomial, eight at a time. Same construction as the
// scalar expFast in SegmenterModel.cpp — 2^n * exp(f) with |f| <= ln2/2 — so the two agree to about
// one part in 10^7, which is where a float's own precision runs out anyway.
inline __m256 exp8(__m256 value) noexcept {
    value = _mm256_max_ps(value, _mm256_set1_ps(-87.0F));
    const __m256i exponent =
        _mm256_cvtps_epi32(_mm256_mul_ps(value, _mm256_set1_ps(1.44269504F)));
    const __m256 wholes = _mm256_cvtepi32_ps(exponent);
    const __m256 f =
        _mm256_fnmadd_ps(wholes, _mm256_set1_ps(0.6931471805599453F), value);

    __m256 poly = _mm256_set1_ps(0.0013888889F);
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.008333334F));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.041666668F));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.16666667F));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(0.5F));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(1.0F));
    poly = _mm256_fmadd_ps(poly, f, _mm256_set1_ps(1.0F));

    const __m256i scale =
        _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(poly, _mm256_castsi256_ps(scale));
}

// An 8x8 float block transposed in registers: eight loads, twenty-four shuffles, eight stores. The
// scalar alternative reads eight floats from eight rows a full source stride apart and writes them
// down a column, so every store lands in a different cache line and the line has to be read before it
// can be written. This does the same work with contiguous loads and contiguous stores.
inline void transpose8x8(const float *source, int sourceStride, float *destination,
                         int destinationStride) noexcept {
    const __m256 r0 = _mm256_loadu_ps(source);
    const __m256 r1 = _mm256_loadu_ps(source + sourceStride);
    const __m256 r2 = _mm256_loadu_ps(source + 2 * sourceStride);
    const __m256 r3 = _mm256_loadu_ps(source + 3 * sourceStride);
    const __m256 r4 = _mm256_loadu_ps(source + 4 * sourceStride);
    const __m256 r5 = _mm256_loadu_ps(source + 5 * sourceStride);
    const __m256 r6 = _mm256_loadu_ps(source + 6 * sourceStride);
    const __m256 r7 = _mm256_loadu_ps(source + 7 * sourceStride);

    const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    _mm256_storeu_ps(destination, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(destination + destinationStride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(destination + 2 * destinationStride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(destination + 3 * destinationStride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(destination + 4 * destinationStride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(destination + 5 * destinationStride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(destination + 6 * destinationStride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(destination + 7 * destinationStride, _mm256_permute2f128_ps(s3, s7, 0x31));
}

float dot(const float *a, const float *b, int count) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int at = 0;
    for (; at + 16 <= count; at += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + at), _mm256_loadu_ps(b + at), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + at + 8), _mm256_loadu_ps(b + at + 8), acc1);
    }
    for (; at + 8 <= count; at += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + at), _mm256_loadu_ps(b + at), acc0);
    float sum = horizontalSum(_mm256_add_ps(acc0, acc1));
    for (; at < count; ++at) sum += a[at] * b[at];
    return sum;
}

void addScaled(float *destination, const float *source, float weight, int count) noexcept {
    const __m256 factor = _mm256_set1_ps(weight);
    int at = 0;
    for (; at + 8 <= count; at += 8) {
        _mm256_storeu_ps(destination + at,
                         _mm256_fmadd_ps(_mm256_loadu_ps(source + at), factor,
                                         _mm256_loadu_ps(destination + at)));
    }
    for (; at < count; ++at) destination[at] += source[at] * weight;
}

void addInto(float *destination, const float *source, int count) noexcept {
    int at = 0;
    for (; at + 8 <= count; at += 8) {
        _mm256_storeu_ps(destination + at, _mm256_add_ps(_mm256_loadu_ps(destination + at),
                                                         _mm256_loadu_ps(source + at)));
    }
    for (; at < count; ++at) destination[at] += source[at];
}

void layerNorm(const float *input, int width, const float *weight, const float *bias,
               float *output) noexcept {
    // Pass one: the mean.
    __m256 total = _mm256_setzero_ps();
    int at = 0;
    for (; at + 8 <= width; at += 8) total = _mm256_add_ps(total, _mm256_loadu_ps(input + at));
    float sum = horizontalSum(total);
    for (; at < width; ++at) sum += input[at];
    const float mean = sum / static_cast<float>(width);

    // Pass two: the variance about it.
    const __m256 centre = _mm256_set1_ps(mean);
    __m256 squares = _mm256_setzero_ps();
    at = 0;
    for (; at + 8 <= width; at += 8) {
        const __m256 centred = _mm256_sub_ps(_mm256_loadu_ps(input + at), centre);
        squares = _mm256_fmadd_ps(centred, centred, squares);
    }
    float variance = horizontalSum(squares);
    for (; at < width; ++at) {
        const float centred = input[at] - mean;
        variance += centred * centred;
    }
    variance /= static_cast<float>(width);
    const float inverse = 1.0F / std::sqrt(variance + NormEpsilon);

    // Pass three: scale, weight, bias.
    const __m256 factor = _mm256_set1_ps(inverse);
    at = 0;
    for (; at + 8 <= width; at += 8) {
        const __m256 centred = _mm256_sub_ps(_mm256_loadu_ps(input + at), centre);
        _mm256_storeu_ps(output + at,
                         _mm256_fmadd_ps(_mm256_mul_ps(centred, factor),
                                         _mm256_loadu_ps(weight + at),
                                         _mm256_loadu_ps(bias + at)));
    }
    for (; at < width; ++at)
        output[at] = (input[at] - mean) * inverse * weight[at] + bias[at];
}

// GELU in the tanh form, which is x * sigmoid(2u) with u = sqrt(2/pi)(x + 0.044715 x^3), so one
// vectorised exp and one divide replace a table lookup with a branch per element.
//
// It is the tanh APPROXIMATION, not the erf GELU the model trained with: they differ by up to about
// 1.5e-3 around |x| = 2.3. Int8 activations are quantised in steps of about 1/127, so this sits well
// under the error already present — but it is a real difference, and why this table is opt-in.
inline __m256 gelu8(__m256 value) noexcept {
    const __m256 cubed = _mm256_mul_ps(_mm256_mul_ps(value, value), value);
    const __m256 inner = _mm256_mul_ps(
        _mm256_set1_ps(0.7978845608F),
        _mm256_fmadd_ps(cubed, _mm256_set1_ps(0.044715F), value));
    const __m256 denominator =
        _mm256_add_ps(_mm256_set1_ps(1.0F), exp8(_mm256_mul_ps(inner, _mm256_set1_ps(-2.0F))));
    return _mm256_div_ps(value, denominator);
}

// GELU as the model actually trained it: 0.5x(1 + erf(x/sqrt2)), with erf from Abramowitz & Stegun
// 7.1.26 - t = 1/(1+px), a degree-5 polynomial in t, times exp(-x^2). Worst error about 1.5e-7, which
// is at the edge of what a float can hold, so this table changes no answers.
//
// It costs one more division and a five-term polynomial over the tanh form. Whether that is worth
// paying is what comparesSpeedOptions measures.
inline __m256 erfGelu8(__m256 value) noexcept {
    const __m256 signMask = _mm256_set1_ps(-0.0F);
    const __m256 sign = _mm256_and_ps(value, signMask);
    const __m256 x = _mm256_mul_ps(_mm256_andnot_ps(signMask, value),
                                   _mm256_set1_ps(0.70710678118654752F)); // |x| / sqrt 2

    const __m256 t = _mm256_div_ps(
        _mm256_set1_ps(1.0F),
        _mm256_fmadd_ps(x, _mm256_set1_ps(0.3275911F), _mm256_set1_ps(1.0F)));
    __m256 poly = _mm256_set1_ps(1.061405429F);
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-1.453152027F));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.421413741F));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-0.284496736F));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.254829592F));
    poly = _mm256_mul_ps(poly, t);

    // erf(|x|) = 1 - poly * exp(-x^2); the sign of x is put back by XOR, since erf is odd.
    const __m256 magnitude = _mm256_sub_ps(
        _mm256_set1_ps(1.0F),
        _mm256_mul_ps(poly, exp8(_mm256_mul_ps(_mm256_mul_ps(x, x), _mm256_set1_ps(-1.0F)))));
    const __m256 erf = _mm256_xor_ps(magnitude, sign);
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5F), value),
                         _mm256_add_ps(_mm256_set1_ps(1.0F), erf));
}

inline float erfGelu1(float value) noexcept {
    return 0.5F * value * (1.0F + std::erf(value * 0.70710678118654752F));
}

void erfGelu(float *values, int count) noexcept {
    int at = 0;
    for (; at + 8 <= count; at += 8)
        _mm256_storeu_ps(values + at, erfGelu8(_mm256_loadu_ps(values + at)));
    for (; at < count; ++at) values[at] = erfGelu1(values[at]);
}

inline float gelu1(float value) noexcept {
    const float inner = 0.7978845608F * (value + 0.044715F * value * value * value);
    return value / (1.0F + std::exp(-2.0F * inner));
}

void gelu(float *values, int count) noexcept {
    int at = 0;
    for (; at + 8 <= count; at += 8)
        _mm256_storeu_ps(values + at, gelu8(_mm256_loadu_ps(values + at)));
    for (; at < count; ++at) values[at] = gelu1(values[at]);
}

void softmax(float *values, int count) noexcept {
    // The largest element, for the usual overflow shift.
    int at = 0;
    __m256 peak = _mm256_set1_ps(values[0]);
    for (; at + 8 <= count; at += 8) peak = _mm256_max_ps(peak, _mm256_loadu_ps(values + at));
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, peak);
    float largest = lanes[0];
    for (int lane = 1; lane < 8; ++lane) largest = std::fmax(largest, lanes[lane]);
    for (; at < count; ++at) largest = std::fmax(largest, values[at]);

    // Exponentiate and total in one pass.
    const __m256 shift = _mm256_set1_ps(largest);
    __m256 total = _mm256_setzero_ps();
    at = 0;
    for (; at + 8 <= count; at += 8) {
        const __m256 exponentiated = exp8(_mm256_sub_ps(_mm256_loadu_ps(values + at), shift));
        _mm256_storeu_ps(values + at, exponentiated);
        total = _mm256_add_ps(total, exponentiated);
    }
    float sum = horizontalSum(total);
    for (; at < count; ++at) {
        values[at] = std::exp(values[at] - largest);
        sum += values[at];
    }

    const __m256 inverse = _mm256_set1_ps(1.0F / sum);
    const float scalarInverse = 1.0F / sum;
    at = 0;
    for (; at + 8 <= count; at += 8)
        _mm256_storeu_ps(values + at, _mm256_mul_ps(_mm256_loadu_ps(values + at), inverse));
    for (; at < count; ++at) values[at] *= scalarInverse;
}

// Scratch for one head: the transposed keys, and the whole score matrix. Thread-local, grown once,
// and deliberately never destroyed - freeing thread_local storage at thread exit corrupts the heap on
// this toolchain, as SegmenterModel.cpp records at length.
thread_local std::vector<float> &keyTransposeStorage = *new std::vector<float>;
thread_local std::vector<float> &scoreMatrixStorage = *new std::vector<float>;
thread_local std::vector<float> &valuePackStorage = *new std::vector<float>;

// ONE CALL FOR A WHOLE LAYER - see the comment on AttendFunction in the header.
//
// The history matters for reading this: it began as one call per (query, key) dot product (no
// faster - the indirection cost what it dispatched), became one call per head (3.2x on the stage via
// key transposition and value packing), and became one call per layer when measurement showed the
// per-call setup at 96% of a small name's attention cost. Each step was driven by a measured gap,
// not a hunch; the comments on the pieces below carry the individual reasons.
void attendHeads(const float *queries, const float *keys, const float *values, int stride, int rows,
                 int headWidth, int heads, float scale, bool fastSoftmax, float *scores,
                 float *context, int contextStride) noexcept {
    const int width = headWidth * heads;
    // Padded so a whole vector of keys is always in bounds. The padding scores get computed and then
    // ignored: nothing downstream looks past `rows`.
    const int keyStride = (rows + 7) / 8 * 8;

    // THE WHOLE K BLOCK TRANSPOSED IN ONE PASS, all heads: [dimension][key] over every dimension of
    // every head, so head h's rows are simply [h*headWidth, (h+1)*headWidth). A dot product in SIMD
    // ends in a horizontal sum - a five-instruction fold per (query, key) pair - and the transpose
    // removes every one: a broadcast of q[d] times eight consecutive keys accumulates eight scores.
    auto &transposed = keyTransposeStorage;
    transposed.resize(static_cast<std::size_t>(width) * keyStride);
    const int wholeKeys = rows / 8 * 8;
    const int wholeWidth = width / 8 * 8;
    for (int key = 0; key < wholeKeys; key += 8) {
        for (int at = 0; at < wholeWidth; at += 8) {
            transpose8x8(keys + static_cast<std::size_t>(key) * stride + at, stride,
                         transposed.data() + static_cast<std::size_t>(at) * keyStride + key,
                         keyStride);
        }
    }
    for (int at = 0; at < width; ++at) {
        float *row = transposed.data() + static_cast<std::size_t>(at) * keyStride;
        const int from = at < wholeWidth ? wholeKeys : 0;
        for (int key = from; key < rows; ++key)
            row[key] = keys[static_cast<std::size_t>(key) * stride + at];
        for (int key = rows; key < keyStride; ++key) row[key] = 0.0F;
    }

    // THE WHOLE V BLOCK PACKED IN ONE PASS: contiguous rows of `width`, so the accumulation below
    // reads sequentially instead of hopping the interleaved projection stride per key.
    auto &packedValues = valuePackStorage;
    packedValues.resize(static_cast<std::size_t>(rows) * width);
    for (int key = 0; key < rows; ++key) {
        std::memcpy(packedValues.data() + static_cast<std::size_t>(key) * width,
                    values + static_cast<std::size_t>(key) * stride,
                    static_cast<std::size_t>(width) * sizeof(float));
    }

    auto &matrix = scoreMatrixStorage;
    matrix.resize(static_cast<std::size_t>(rows) * keyStride);
    const __m256 scaleVector = _mm256_set1_ps(scale);

    for (int head = 0; head < heads; ++head) {
        const int offset = head * headWidth;
        const float *keyRows = transposed.data() + static_cast<std::size_t>(offset) * keyStride;

        // Phase one: the score matrix, four queries by eight keys at a time - four independent FMA
        // chains, and each key vector loaded once per four queries.
        int query = 0;
        for (; query + 4 <= rows; query += 4) {
            const float *q0 = queries + static_cast<std::size_t>(query) * stride + offset;
            const float *q1 = q0 + stride;
            const float *q2 = q1 + stride;
            const float *q3 = q2 + stride;
            for (int keyBlock = 0; keyBlock < keyStride; keyBlock += 8) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                for (int at = 0; at < headWidth; ++at) {
                    const __m256 keyLane = _mm256_loadu_ps(
                        keyRows + static_cast<std::size_t>(at) * keyStride + keyBlock);
                    a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(q0 + at), keyLane, a0);
                    a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(q1 + at), keyLane, a1);
                    a2 = _mm256_fmadd_ps(_mm256_broadcast_ss(q2 + at), keyLane, a2);
                    a3 = _mm256_fmadd_ps(_mm256_broadcast_ss(q3 + at), keyLane, a3);
                }
                float *out = matrix.data() + static_cast<std::size_t>(query) * keyStride + keyBlock;
                _mm256_storeu_ps(out, _mm256_mul_ps(a0, scaleVector));
                _mm256_storeu_ps(out + keyStride, _mm256_mul_ps(a1, scaleVector));
                _mm256_storeu_ps(out + 2 * keyStride, _mm256_mul_ps(a2, scaleVector));
                _mm256_storeu_ps(out + 3 * keyStride, _mm256_mul_ps(a3, scaleVector));
            }
        }
        for (; query < rows; ++query) {
            const float *queryVector = queries + static_cast<std::size_t>(query) * stride + offset;
            for (int keyBlock = 0; keyBlock < keyStride; keyBlock += 8) {
                __m256 accumulator = _mm256_setzero_ps();
                for (int at = 0; at < headWidth; ++at) {
                    accumulator = _mm256_fmadd_ps(
                        _mm256_broadcast_ss(queryVector + at),
                        _mm256_loadu_ps(keyRows + static_cast<std::size_t>(at) * keyStride +
                                        keyBlock),
                        accumulator);
                }
                _mm256_storeu_ps(matrix.data() + static_cast<std::size_t>(query) * keyStride +
                                     keyBlock,
                                 _mm256_mul_ps(accumulator, scaleVector));
            }
        }

        // Phase two: normalise every row before touching the values, so the value block below is
        // walked once per four queries rather than interleaved with a softmax each time.
        for (int at = 0; at < rows; ++at) {
            float *row = matrix.data() + static_cast<std::size_t>(at) * keyStride;
            if (fastSoftmax) {
                softmax(row, rows);
            } else {
                // The float32 reference precision keeps the library exp; only the shift is shared.
                float largest = row[0];
                for (int key = 1; key < rows; ++key) largest = std::fmax(largest, row[key]);
                float total = 0.0F;
                for (int key = 0; key < rows; ++key) {
                    row[key] = std::exp(row[key] - largest);
                    total += row[key];
                }
                const float inverse = 1.0F / total;
                for (int key = 0; key < rows; ++key) row[key] *= inverse;
            }
        }
        (void)scores; // the caller's row-sized scratch; the matrix above replaced it

        // Phase three: context = weights * values, four queries at a time so each value vector is
        // loaded once for four accumulations. Values for this head sit at `offset` in each packed
        // row.
        int out = 0;
        for (; out + 4 <= rows; out += 4) {
            const float *w0 = matrix.data() + static_cast<std::size_t>(out) * keyStride;
            const float *w1 = w0 + keyStride;
            const float *w2 = w1 + keyStride;
            const float *w3 = w2 + keyStride;
            float *c0 = context + static_cast<std::size_t>(out) * contextStride + offset;
            int lane = 0;
            for (; lane + 8 <= headWidth; lane += 8) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                for (int key = 0; key < rows; ++key) {
                    const __m256 value = _mm256_loadu_ps(
                        packedValues.data() + static_cast<std::size_t>(key) * width + offset +
                        lane);
                    a0 = _mm256_fmadd_ps(value, _mm256_broadcast_ss(w0 + key), a0);
                    a1 = _mm256_fmadd_ps(value, _mm256_broadcast_ss(w1 + key), a1);
                    a2 = _mm256_fmadd_ps(value, _mm256_broadcast_ss(w2 + key), a2);
                    a3 = _mm256_fmadd_ps(value, _mm256_broadcast_ss(w3 + key), a3);
                }
                _mm256_storeu_ps(c0 + lane, a0);
                _mm256_storeu_ps(c0 + contextStride + lane, a1);
                _mm256_storeu_ps(c0 + 2 * contextStride + lane, a2);
                _mm256_storeu_ps(c0 + 3 * contextStride + lane, a3);
            }
            for (; lane < headWidth; ++lane) {
                float s0 = 0.0F, s1 = 0.0F, s2 = 0.0F, s3 = 0.0F;
                for (int key = 0; key < rows; ++key) {
                    const float value =
                        packedValues[static_cast<std::size_t>(key) * width + offset + lane];
                    s0 += value * w0[key];
                    s1 += value * w1[key];
                    s2 += value * w2[key];
                    s3 += value * w3[key];
                }
                c0[lane] = s0;
                c0[contextStride + lane] = s1;
                c0[2 * contextStride + lane] = s2;
                c0[3 * contextStride + lane] = s3;
            }
        }
        for (; out < rows; ++out) {
            const float *row = matrix.data() + static_cast<std::size_t>(out) * keyStride;
            float *destination = context + static_cast<std::size_t>(out) * contextStride + offset;
            int lane = 0;
            for (; lane + 8 <= headWidth; lane += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (int key = 0; key < rows; ++key) {
                    sum = _mm256_fmadd_ps(
                        _mm256_loadu_ps(packedValues.data() +
                                        static_cast<std::size_t>(key) * width + offset + lane),
                        _mm256_broadcast_ss(row + key), sum);
                }
                _mm256_storeu_ps(destination + lane, sum);
            }
            for (; lane < headWidth; ++lane) {
                float sum = 0.0F;
                for (int key = 0; key < rows; ++key)
                    sum += packedValues[static_cast<std::size_t>(key) * width + offset + lane] *
                           row[key];
                destination[lane] = sum;
            }
        }
    }
}

const Table accurate{&dot,     &addScaled, &addInto,    &layerNorm,
                     &erfGelu, &softmax,   &attendHeads};
const Table approximate{&dot,  &addScaled, &addInto,    &layerNorm,
                        &gelu, &softmax,   &attendHeads};

} // namespace

const Table *avx2Table() noexcept { return &accurate; }

const Table *avx2FastGeluTable() noexcept { return &approximate; }

#else

const Table *avx2Table() noexcept { return nullptr; }

const Table *avx2FastGeluTable() noexcept { return nullptr; }

#endif

} // namespace neurelease::model::floats
