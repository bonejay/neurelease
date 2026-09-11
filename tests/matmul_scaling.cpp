// How much does kernels::matmul gain from more activation rows, at the shapes this model uses?
// The token transformer runs at M = tokens (median 10 in the corpus); cross-name batching would
// raise that to 4-8x. This measures whether that is worth building.
#include "model/kernels/kernels.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace neurelease::model;

static double runShape(int rows, int in, int m, int repeats) {
    kernels::PackedMatrix packed;
    std::vector<std::int8_t> weights(static_cast<std::size_t>(rows) * in);
    std::vector<float> scales(rows, 0.01F);
    std::mt19937 rng(7);
    for (auto& w : weights) w = static_cast<std::int8_t>(rng() % 255 - 127);
    packed.pack(weights.data(), scales.data(), rows, in);
    const int stride = packed.stride;
    // THE ACTIVATION WIDTH IS THE PATH'S, NOT A CHOICE: the AVX2 path reads int16 codes, and an
    // int8 buffer here is a half-length read straight off the end of the allocation.
    const bool wide = kernels::activationKind(kernels::activePath()) == kernels::ActivationKind::Int16;
    std::vector<std::int16_t> acts16(wide ? static_cast<std::size_t>(m) * stride : 0, 3);
    std::vector<std::int8_t> acts8(wide ? 0 : static_cast<std::size_t>(m) * stride, 3);
    const void* acts = wide ? static_cast<const void*>(acts16.data())
                            : static_cast<const void*>(acts8.data());
    std::vector<std::int32_t> out(static_cast<std::size_t>(m) * rows);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i)
        kernels::matmul(packed, acts, m, stride, out.data());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double macs = double(repeats) * m * rows * in;
    return macs / seconds / 1e9;   // GMAC/s
}

int main() {
    struct Shape { const char* what; int rows; int in; };
    const Shape shapes[] = {
        {"transformer qkv/out 256x256", 256, 256},
        {"transformer ff1 384x256", 384, 256},
        {"transformer ff2 256x384", 256, 384},
        {"char conv 144x720 (k5*144ch)", 144, 720},
    };
    printf("%-30s", "shape");
    for (int m : {1, 5, 10, 20, 40, 80, 160}) printf("  M=%-3d", m);
    printf("\n");
    for (const Shape& s : shapes) {
        printf("%-30s", s.what);
        for (int m : {1, 5, 10, 20, 40, 80, 160}) {
            const int repeats = 20000 / m + 50;
            printf("  %5.1f", runShape(s.rows, s.in, m, repeats));
        }
        printf("   GMAC/s\n");
    }
    return 0;
}
