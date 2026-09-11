// The float side of the forward pass, as a runtime-swappable table.
//
// The int8 matmuls have had hand-written SIMD per instruction set since the beginning. Everything
// AROUND them did not: layer norm, GELU, the attention dot products and the value accumulation were
// plain loops with, at best, an SSE2 hand-vectorisation — 4 lanes wide, on a machine that has had 8
// since 2013. They are not the majority of the time, but they are not nothing either: attention alone
// is about a tenth of it, and layer norm, GELU and the residual adds are most of what is left over
// after the matmuls and convolutions.
//
// The same structure as the kernels: one translation unit per instruction set, compiled with exactly
// its own flags, chosen at RUNTIME. The library still runs on a CPU without AVX2 — it just uses the
// baseline table.
//
// NOT bit-identical. The AVX2 GELU uses the tanh form rather than the exact erf, and its exp is a
// polynomial, so results differ in the last few digits of a float. Against int8 activation
// quantisation — one part in 127 — that is far below the noise the model already carries, but it is a
// difference, so it is opt-in and measured (see comparesSpeedOptions).

#ifndef NEURELEASE_SEGMENTER_FLOATS_H
#define NEURELEASE_SEGMENTER_FLOATS_H

namespace neurelease::model::floats {

constexpr float NormEpsilon = 1e-5F; // torch.nn.LayerNorm default, shared by ChannelNorm

using DotFunction = float (*)(const float *, const float *, int) noexcept;
using AddScaledFunction = void (*)(float *, const float *, float, int) noexcept;
using AddIntoFunction = void (*)(float *, const float *, int) noexcept;
using LayerNormFunction = void (*)(const float *, int, const float *, const float *,
                                   float *) noexcept;
using MapFunction = void (*)(float *, int) noexcept;

// ONE CALL FOR A WHOLE ATTENTION HEAD, not one per dot product.
//
// The first version of this table exposed `dot` and `addScaled` and let the model call them from
// inside the O(rows^2) loops. That made attention barely faster: the operations are 32 floats long, so
// an indirect call that also blocks inlining costs about as much as the work it dispatches. Measured,
// attention went from 178 to 173 microseconds per name - nothing.
//
// So the whole head comes through here instead: scores, softmax and the value accumulation for every
// query, one call. `queries`, `keys` and `values` each point at this head's slice, `stride` floats
// apart (the projection interleaves q, k and v, so the stride is not the head width), `scores` is
// scratch for `rows` floats, and `context` is where each query's output lands, `contextStride` apart.
// ONE CALL PER LAYER, ALL HEADS - because the per-call overhead, not the arithmetic, is where
// attention's time went. Measured on the real model (8 heads, headWidth 32, 4 layers): a head call
// at 4 tokens costs 1.59 us in situ while its arithmetic floor is ~0.06 us - the other 96% is the
// transpose, the value packing, the scratch bookkeeping and the call itself, and it was being paid
// 32 times per name. Attending every head in one call pays it 4 times instead, and the transpose
// and packing loops cover the whole K and V blocks in one pass rather than 8 strided passes.
//
// `queries`/`keys`/`values` point at the START of each region (the projection interleaves them,
// `stride` floats apart per token); head h reads the slice [h*headWidth, (h+1)*headWidth) of each
// and writes its context at the same offset, `contextStride` apart per token.
using AttendFunction = void (*)(const float *queries, const float *keys, const float *values,
                                int stride, int rows, int headWidth, int heads, float scale,
                                bool fastSoftmax, float *scores, float *context,
                                int contextStride) noexcept;

// Every member is non-null in a usable table. `gelu` and `softmax` are the FAST forms only — the
// float32 reference precision keeps std::erf and std::exp and does not come through here.
struct Table {
    DotFunction dot = nullptr;
    AddScaledFunction addScaled = nullptr;
    AddIntoFunction addInto = nullptr;
    LayerNormFunction layerNorm = nullptr;
    MapFunction gelu = nullptr;
    MapFunction softmax = nullptr;
    AttendFunction attendHeads = nullptr;
};

// TWO AVX2 TABLES, differing only in GELU.
//
// `avx2Table` uses an erf GELU accurate to about one part in 10^7 - the same function the model
// trained with, so switching to it does not move the answers. `avx2FastGeluTable` uses the tanh
// approximation instead: measurably quicker, and off by up to 1.5e-3 around |x| = 2.3, which is
// enough to change the parse of about 1% of names. Both return nullptr where this build has no AVX2
// translation unit. Whether the CPU can run it is the caller's question, answered with the kernel
// dispatcher's CPUID result.
const Table *avx2Table() noexcept;
const Table *avx2FastGeluTable() noexcept;

} // namespace neurelease::model::floats

#endif
