#include "model/kernels/kernels.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define NEURELEASE_X86_64 1
#include <emmintrin.h> // SSE2, baseline on every x86-64
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace neurelease::model::kernels {

namespace {


#ifdef NEURELEASE_X86_64

struct CpuidResult {
    std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuidResult cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
    CpuidResult result;
#if defined(_MSC_VER)
    int registers[4];
    __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
    result = {static_cast<std::uint32_t>(registers[0]), static_cast<std::uint32_t>(registers[1]),
              static_cast<std::uint32_t>(registers[2]), static_cast<std::uint32_t>(registers[3])};
#else
    __get_cpuid_count(leaf, subleaf, &result.eax, &result.ebx, &result.ecx, &result.edx);
#endif
    return result;
}

std::uint64_t xcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    std::uint32_t low = 0, high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<std::uint64_t>(high) << 32) | low;
#endif
}

Path detect() noexcept {
    const auto basic = cpuid(0, 0);
    if (basic.eax < 7) {
        return Path::Scalar;
    }
    const auto one = cpuid(1, 0);
    const bool osxsave = (one.ecx & (1U << 27)) != 0;
    const bool avx = (one.ecx & (1U << 28)) != 0;
    if (!osxsave || !avx) {
        return Path::Scalar;
    }
    const auto state = xcr0();
    const bool ymmEnabled = (state & 0x6) == 0x6;
    if (!ymmEnabled) {
        return Path::Scalar;
    }
    const auto seven = cpuid(7, 0);
    const bool avx2 = (seven.ebx & (1U << 5)) != 0;
    if (!avx2) {
        return Path::Scalar;
    }
    const bool avx512f = (seven.ebx & (1U << 16)) != 0;
    const bool avx512bw = (seven.ebx & (1U << 30)) != 0;
    const bool avx512vl = (seven.ebx & (1U << 31)) != 0;
    const bool avx512vnni = (seven.ecx & (1U << 11)) != 0;
    const bool zmmEnabled = (state & 0xE6) == 0xE6;
    if (avx512f && avx512bw && avx512vl && avx512vnni && zmmEnabled &&
        avx512VnniMatvec() != nullptr) {
        return Path::Avx512Vnni;
    }
    const auto sevenOne = cpuid(7, 1);
    const bool avxVnni = (sevenOne.eax & (1U << 4)) != 0;
    if (avxVnni && avx2VnniMatvec() != nullptr) {
        return Path::Avx2Vnni;
    }
    return avx2Matvec() != nullptr ? Path::Avx2 : Path::Scalar;
}

#else

Path detect() noexcept { return Path::Scalar; }

#endif // NEURELEASE_X86_64

// Whether the fast, saturating kernels apply. Only the plain AVX2 path has anything to gain: the
// VNNI and AVX-512 instructions already take unsigned activations and accumulate into int32, so they
// are both faster AND exact, and scalar has nothing to swap in.
// The default zero point: 128, i.e. full-range activations. Measured over 2,000 real names, dropping
// to 64 to make saturation impossible changed 1.80% of parses where 128 changes 1.10% - the lost
// activation resolution costs more than the rare overflow it prevents. See the sweep in
// comparesSpeedOptions.
std::atomic<int> fastZeroPointValue{128};

bool zeroPointSupported(int zeroPoint) noexcept {
    return zeroPoint == 64 || zeroPoint == 80 || zeroPoint == 96 || zeroPoint == 112 ||
           zeroPoint == 128;
}

bool fastApplies(Path path, Accuracy accuracy) noexcept {
    return accuracy != Accuracy::Exact && path == Path::Avx2 &&
           avx2FastMatvec(fastZeroPointValue.load(std::memory_order_relaxed)) != nullptr;
}

MatvecFunction functionFor(Path path) noexcept {
    switch (path) {
    case Path::Avx512Vnni:
        return avx512VnniMatvec();
    case Path::Avx2Vnni:
        return avx2VnniMatvec();
    case Path::Avx2:
        return avx2Matvec();
    case Path::Scalar:
        break;
    }
    return scalarMatvec();
}

Path capFromEnvironment(Path detected) noexcept {
    const char *requested = std::getenv("NEURELEASE_SEGMENTER_KERNEL");
    if (requested == nullptr) {
        return detected;
    }
    const std::string name(requested);
    const Path all[] = {Path::Scalar, Path::Avx2, Path::Avx2Vnni, Path::Avx512Vnni};
    for (const auto candidate : all) {
        // A cap, not a promise: asking for a path the CPU lacks falls back to the best real one
        // rather than crashing on an illegal instruction.
        if (name == pathName(candidate)) {
            return candidate <= detected ? candidate : detected;
        }
    }
    return detected;
}

std::atomic<MatvecFunction> activeFunction{nullptr};
std::atomic<MatmulFunction> activeMatmul{nullptr};
std::atomic<Path> activePathValue{Path::Scalar};
std::atomic<Accuracy> accuracyValue{Accuracy::Exact};
std::atomic<bool> demotedFlag{false};
std::atomic<Path> refusedPath{Path::Scalar};

// Set once before first use; only read under resolveMutex thereafter.
std::mutex resolveMutex;
std::string badPathFile;

// The persistence file holds two kinds of lines. "<path>" is a confirmed-bad path (it failed the
// known-answer check). "trying <path>" is a crash marker: it is written and flushed BEFORE the
// first instruction of a SIMD path ever executes and removed right after the self-check returns
// — so if the process dies in between (an illegal instruction on a machine whose feature
// reporting lies, a driver-level fault), the marker survives, and every later run treats that
// path as bad without ever executing it again. Both read as banned.
bool isRecordedBad(Path path) {
    if (badPathFile.empty()) {
        return false;
    }
    std::ifstream stream(badPathFile);
    std::string line;
    const std::string name = pathName(path);
    while (std::getline(stream, line)) {
        if (line == name || line == "trying " + name) {
            return true;
        }
    }
    return false;
}

void recordBad(Path path) {
    if (badPathFile.empty()) {
        return;
    }
    std::ofstream stream(badPathFile, std::ios::app);
    stream << pathName(path) << '\n';
}

void markTrying(Path path) {
    if (badPathFile.empty()) {
        return;
    }
    std::ofstream stream(badPathFile, std::ios::app);
    stream << "trying " << pathName(path) << '\n';
    stream.flush();
}

void clearTrying(Path path) {
    if (badPathFile.empty()) {
        return;
    }
    std::ifstream input(badPathFile);
    std::string content, line;
    const std::string marker = std::string("trying ") + pathName(path);
    while (std::getline(input, line)) {
        if (!line.empty() && line != marker) {
            content += line;
            content += '\n';
        }
    }
    input.close();
    std::ofstream(badPathFile, std::ios::trunc) << content;
}

// Known-answer check: the candidate's kernels against the scalar path, on synthetic codes that
// cover the extremes (+/-127, zero padding, the +128 bias corner). Runs in microseconds, once
// per process, and is what makes "any SIMD path is bit-identical to scalar" a checked property
// on every machine this ever runs on rather than only on the machines we tested.
bool selfCheck(Path candidate) noexcept {
    const MatvecFunction function = functionFor(candidate);
    const bool wide = activationKind(candidate) == ActivationKind::Int16;
    constexpr int Rows = 5, Width = 192, ActCount = 5, ActStride = 64;
    constexpr int Region = (ActCount - 1) * ActStride + Width;

    std::vector<std::int8_t> codes(static_cast<std::size_t>(Rows) * Width);
    for (std::size_t at = 0; at < codes.size(); ++at) {
        codes[at] = static_cast<std::int8_t>(static_cast<int>((at * 89 + 17) % 255) - 127);
    }
    codes[0] = 127;
    codes[1] = -127;
    std::vector<float> scales(Rows, 1.0F);
    PackedMatrix matrix;
    matrix.pack(codes.data(), scales.data(), Rows, Width);

    std::vector<std::int8_t> acts8(Region + 64, 0);
    std::vector<std::int16_t> acts16(Region + 64, 0);
    for (int at = 0; at < Region; ++at) {
        const auto value = static_cast<std::int8_t>((at * 37 + 5) % 255 - 127);
        acts8[static_cast<std::size_t>(at)] = value;
        acts16[static_cast<std::size_t>(at)] = value;
    }
    acts8[0] = -127;
    acts16[0] = -127;
    acts8[1] = 127;
    acts16[1] = 127;

    const auto actAt = [&](int act) -> const void * {
        const std::size_t offset = static_cast<std::size_t>(act) * ActStride;
        return wide ? static_cast<const void *>(acts16.data() + offset)
                    : static_cast<const void *>(acts8.data() + offset);
    };

    std::int32_t expected[ActCount][Rows];
    for (int act = 0; act < ActCount; ++act) {
        scalarMatvec()(matrix, acts8.data() + static_cast<std::size_t>(act) * ActStride,
                       expected[act]);
    }

    std::int32_t got[Rows];
    for (int act = 0; act < ActCount; ++act) {
        function(matrix, actAt(act), got);
        if (std::memcmp(got, expected[act], sizeof(got)) != 0) {
            return false;
        }
    }

    MatmulFunction blocked = nullptr;
    switch (candidate) {
    case Path::Avx512Vnni:
        blocked = avx512VnniMatmul();
        break;
    case Path::Avx2Vnni:
        blocked = avx2VnniMatmul();
        break;
    case Path::Avx2:
        blocked = avx2Matmul();
        break;
    case Path::Scalar:
        break;
    }
    if (blocked != nullptr) {
        std::int32_t batch[ActCount * Rows];
        blocked(matrix, wide ? static_cast<const void *>(acts16.data())
                             : static_cast<const void *>(acts8.data()),
                ActCount, ActStride, batch);
        for (int act = 0; act < ActCount; ++act) {
            if (std::memcmp(batch + act * Rows, expected[act], sizeof(expected[act])) != 0) {
                return false;
            }
        }
    }
    return true;
}

// The generic batched form: every path without a hand-blocked matmul just walks its matvec.
// Only ever called through activeMatmul, so the dispatch state is already resolved here.
void loopedMatmul(const PackedMatrix &matrix, const void *activations, int actCount,
                  int actStride, std::int32_t *accumulators) {
    const auto function = activeFunction.load(std::memory_order_relaxed);
    const auto kind = activationKind(activePathValue.load(std::memory_order_relaxed));
    const int elementBytes = kind == ActivationKind::Int16 ? 2 : 1;
    for (int act = 0; act < actCount; ++act) {
        function(matrix,
                 static_cast<const char *>(activations) +
                     static_cast<std::size_t>(act) * actStride * elementBytes,
                 accumulators + static_cast<std::size_t>(act) * matrix.rows);
    }
}

MatmulFunction matmulFor(Path path) noexcept {
    MatmulFunction blocked = nullptr;
    switch (path) {
    case Path::Avx512Vnni:
        blocked = avx512VnniMatmul();
        break;
    case Path::Avx2Vnni:
        blocked = avx2VnniMatmul();
        break;
    case Path::Avx2:
        blocked = avx2Matmul();
        break;
    case Path::Scalar:
        break;
    }
    return blocked != nullptr ? blocked : &loopedMatmul;
}

// Points the dispatch at the kernels for (path, accuracy). Takes no lock and does NOT resolve, so
// resolveOnce can call it while already holding the resolve mutex - calling the public setAccuracy
// from in there deadlocks on a non-recursive mutex, which it did.
void applyDispatch(Path path, Accuracy accuracy) noexcept {
    if (fastApplies(path, accuracy)) {
        const int zeroPoint = fastZeroPointValue.load(std::memory_order_relaxed);
        activeMatmul.store(avx2FastMatmul(zeroPoint), std::memory_order_relaxed);
        activeFunction.store(avx2FastMatvec(zeroPoint), std::memory_order_release);
    } else {
        activeMatmul.store(matmulFor(path), std::memory_order_relaxed);
        activeFunction.store(functionFor(path), std::memory_order_release);
    }
}

Accuracy accuracyFromEnvironment() noexcept {
    if (const char *wanted = std::getenv("NEURELEASE_SEGMENTER_ACCURACY")) {
        const std::string text(wanted);
        if (text == "fast") return Accuracy::Fast;
    }
    return Accuracy::Exact;
}

void resolveOnce() noexcept {
    if (activeFunction.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> guard(resolveMutex);
    if (activeFunction.load(std::memory_order_acquire) != nullptr) {
        return;
    }
    auto path = capFromEnvironment(detect());
    if (path != Path::Scalar) {
        if (isRecordedBad(path)) {
            // Either it failed the check once, or a previous run CRASHED between the "trying"
            // marker and its confirmation — treat both as proof and never execute it again.
            demotedFlag.store(true, std::memory_order_relaxed);
            refusedPath.store(path, std::memory_order_relaxed);
            path = Path::Scalar;
        } else {
            markTrying(path);
            const bool healthy = selfCheck(path);
            clearTrying(path);
            if (!healthy) {
                // The 100% safe answer, not the next-fastest: a machine where one SIMD path
                // lies cannot be trusted about the others either.
                demotedFlag.store(true, std::memory_order_relaxed);
                refusedPath.store(path, std::memory_order_relaxed);
                recordBad(path);
                path = Path::Scalar;
            }
        }
    }
    activePathValue.store(path, std::memory_order_relaxed);
    // The accuracy mode is read from the environment on this first resolve and applied here, so the
    // very first parse already runs whichever kernels were asked for.
    if (const char *asked = std::getenv("NEURELEASE_SEGMENTER_ZERO_POINT")) {
        const int zeroPoint = std::atoi(asked);
        if (zeroPointSupported(zeroPoint))
            fastZeroPointValue.store(zeroPoint, std::memory_order_relaxed);
    }
    const Accuracy wanted = accuracyFromEnvironment();
    accuracyValue.store(wanted, std::memory_order_relaxed);
    applyDispatch(path, wanted);
}

// The probe's state. One counter set per thread, merged when read, so measuring costs no
// synchronisation in the inner loop — and it is all behind a flag that is off in every normal run.
struct ProbeCounters {
    long long dotProducts = 0;
    long long pairs = 0;
    long long saturatedPairs = 0;
    long long dotProductsAffected = 0;
    long long absoluteErrorSum = 0;
    int worstAbsoluteError = 0;
};

std::atomic<bool> probeEnabled{false};
std::atomic<bool> probeSubstitute{false};
std::mutex probeMutex;
std::vector<ProbeCounters *> probeRegistry;

ProbeCounters &registerProbeCounters() {
    // Leaked on purpose, like the model scratch: destroying a thread_local at thread exit frees
    // through the wrong heap on this toolchain.
    auto *counters = new ProbeCounters;
    const std::lock_guard<std::mutex> guard(probeMutex);
    probeRegistry.push_back(counters);
    return *counters;
}

thread_local ProbeCounters &probeCounters = registerProbeCounters();

void scalar(const PackedMatrix &matrix, const void *activation, std::int32_t *accumulators) {
    const auto *codes = static_cast<const std::int8_t *>(activation);
    const bool probing = probeEnabled.load(std::memory_order_relaxed);
    const bool substituting = probeSubstitute.load(std::memory_order_relaxed);
    for (int row = 0; row < matrix.rows; ++row) {
        const std::int8_t *weights = matrix.codes.data() +
                                     static_cast<std::size_t>(row) * matrix.stride;
        std::int32_t sum = 0;
        for (int at = 0; at < matrix.width; ++at) {
            sum += static_cast<std::int32_t>(weights[at]) *
                   static_cast<std::int32_t>(codes[at]);
        }
        if (!probing) {
            accumulators[row] = sum;
            continue;
        }

        // THE u8 PATH, SIMULATED EXACTLY AS THE INSTRUCTION WOULD DO IT.
        //
        // Activations become unsigned by adding a zero point of 128 (the standard symmetric-to-
        // unsigned shift), adjacent pairs are multiplied and added in a SIGNED 16-BIT lane with
        // saturation, and the pair results are then widened and summed without further loss —
        // which is VPMADDUBSW followed by VPMADDWD against ones. The zero point is removed at the
        // end: sum(u_i * w_i) - 128 * sum(w_i) is the signed result.
        std::int32_t simulated = 0;
        std::int32_t weightSum = 0;
        long long saturated = 0;
        for (int at = 0; at < matrix.width; at += 2) {
            const int firstWeight = weights[at];
            const int firstUnsigned = static_cast<int>(codes[at]) + 128;
            weightSum += firstWeight;
            std::int32_t pair = firstWeight * firstUnsigned;
            if (at + 1 < matrix.width) {
                const int secondWeight = weights[at + 1];
                const int secondUnsigned = static_cast<int>(codes[at + 1]) + 128;
                weightSum += secondWeight;
                pair += secondWeight * secondUnsigned;
            }
            if (pair > 32767) {
                pair = 32767;
                ++saturated;
            } else if (pair < -32768) {
                pair = -32768;
                ++saturated;
            }
            simulated += pair;
        }
        simulated -= 128 * weightSum;

        ProbeCounters &counters = probeCounters;
        ++counters.dotProducts;
        counters.pairs += (matrix.width + 1) / 2;
        counters.saturatedPairs += saturated;
        const std::int32_t error = simulated - sum;
        if (error != 0) {
            ++counters.dotProductsAffected;
            const int magnitude = error < 0 ? -error : error;
            counters.absoluteErrorSum += magnitude;
            if (magnitude > counters.worstAbsoluteError) counters.worstAbsoluteError = magnitude;
        }
        accumulators[row] = substituting ? simulated : sum;
    }
}

} // namespace

const char *pathName(Path path) noexcept {
    switch (path) {
    case Path::Scalar:
        return "scalar";
    case Path::Avx2:
        return "avx2";
    case Path::Avx2Vnni:
        return "avx2vnni";
    case Path::Avx512Vnni:
        return "avx512vnni";
    }
    return "scalar";
}

int strideQuantum(Path path) noexcept {
    switch (path) {
    case Path::Scalar:
        return 16;  // walks the true width; 16 only keeps the activation buffers tidy
    case Path::Avx2:
    case Path::Avx2Vnni:
        return 32;  // the avx2 tiles step 32 codes at a time; the vnni one has a 32-tail
    case Path::Avx512Vnni:
        return 64;  // steps 64 with no tail loop, so a shorter row would be read past
    }
    return 64;
}

int strideQuantum() noexcept {
    // NEURELEASE_STRIDE_QUANTUM raises it, for measuring what the padding costs: the old flat 64
    // is still reachable with it. Only ever raised, never lowered - a value below what the widest
    // runnable kernel steps would have it read past the end of a row.
    static const int quantum = [] {
        const int required = strideQuantum(detect());
        if (const char *wanted = std::getenv("NEURELEASE_STRIDE_QUANTUM")) {
            const int asked = std::atoi(wanted);
            if (asked > required && asked % required == 0) return asked;
        }
        return required;
    }();
    return quantum;
}

Path activePath() noexcept {
    resolveOnce();
    return activePathValue.load(std::memory_order_relaxed);
}

void setAccuracy(Accuracy wanted) noexcept {
    resolveOnce();
    accuracyValue.store(wanted, std::memory_order_relaxed);
    // No self-check, deliberately: the fast path is KNOWN not to match scalar, so the usual "differs
    // from scalar means broken" rule would demote it on sight. The exact path it replaces was already
    // checked at resolve.
    applyDispatch(activePathValue.load(std::memory_order_relaxed), wanted);
}

Accuracy accuracy() noexcept { return accuracyValue.load(std::memory_order_relaxed); }

void setFastZeroPoint(int zeroPoint) noexcept {
    if (!zeroPointSupported(zeroPoint)) return;
    resolveOnce();
    fastZeroPointValue.store(zeroPoint, std::memory_order_relaxed);
    // Re-point the dispatch: the zero point picks the kernel instantiation as well as the clamp.
    applyDispatch(activePathValue.load(std::memory_order_relaxed),
                  accuracyValue.load(std::memory_order_relaxed));
}

int fastZeroPoint() noexcept { return fastZeroPointValue.load(std::memory_order_relaxed); }

MatmulFunction exactMatmul() noexcept {
    return matmulFor(activePath());
}

ActivationKind exactActivationKind() noexcept {
    return activationKind(activePath());
}

int activationLimit() noexcept {
    // The clamp only narrows where the u8 kernel is actually dispatched; every other path takes the
    // full range. One below the zero point, so the shifted value stays positive.
    const Accuracy mode = accuracyValue.load(std::memory_order_relaxed);
    if (mode != Accuracy::Fast) return 127;
    if (!fastApplies(activePathValue.load(std::memory_order_relaxed), mode)) return 127;
    return fastZeroPointValue.load(std::memory_order_relaxed) - 1;
}

ActivationKind activationKind() noexcept {
    const Path path = activePath();
    // The fast kernel reads activations as unsigned BYTES, not int16 - the whole point of it.
    if (fastApplies(path, accuracyValue.load(std::memory_order_relaxed)))
        return ActivationKind::Int8;
    return activationKind(path);
}

void forcePath(Path path) {
    if (path > detect()) {
        throw std::runtime_error(std::string("segmenter kernels: this CPU cannot run ") +
                                 pathName(path));
    }
    activePathValue.store(path, std::memory_order_relaxed);
    // Re-applies the accuracy mode, so forcing a path never silently reverts it.
    applyDispatch(path, accuracyValue.load(std::memory_order_relaxed));
}

namespace {

std::mutex packingMutex;
PackingStats packingTotals;

void notePacking(int rows, int width, int stride) {
    const std::lock_guard<std::mutex> guard(packingMutex);
    ++packingTotals.matrices;
    packingTotals.widthElements += static_cast<long long>(rows) * width;
    packingTotals.strideElements += static_cast<long long>(rows) * stride;
    const auto roundUp = [width](int quantum) { return (width + quantum - 1) / quantum * quantum; };
    packingTotals.strideElements32 += static_cast<long long>(rows) * roundUp(32);
    packingTotals.strideElements16 += static_cast<long long>(rows) * roundUp(16);
    // The worst OFFENDER is the one with the largest share of padding, not the largest matrix:
    // a 68-wide row padded to 128 wastes 47% of every multiply it takes part in.
    const bool worse = packingTotals.worstStride == 0
                       || double(width) / double(stride)
                              < double(packingTotals.worstWidth) / double(packingTotals.worstStride);
    if (worse) {
        packingTotals.worstWidth = width;
        packingTotals.worstStride = stride;
    }
}

}  // namespace

std::atomic<bool> pairConstraintValue{[] {
    const char *wanted = std::getenv("NEURELEASE_PAIR_CONSTRAINT");
    return wanted != nullptr && wanted[0] == '1';
}()};

void setPairConstraint(bool enabled) noexcept {
    pairConstraintValue.store(enabled, std::memory_order_relaxed);
}

bool pairConstraint() noexcept { return pairConstraintValue.load(std::memory_order_relaxed); }

namespace {

// Shrink one same-sign pair onto |a| + |b| <= 128, cutting both magnitudes equally - the Euclidean
// projection onto the constraint, so the weight vector moves as little as it can. The caller has
// already established the pair shares a sign and exceeds the bound.
void constrainPair(std::int8_t &a, std::int8_t &b, long long &cut) {
    int magnitudeA = a < 0 ? -a : a;
    int magnitudeB = b < 0 ? -b : b;
    int excess = magnitudeA + magnitudeB - 128;
    cut += excess;
    // Equal halves; the odd unit comes off the larger magnitude, and a magnitude never crosses zero
    // (the bound is 128, so at least one magnitude is >= 65 and survives its half of the cut).
    int cutA = excess / 2;
    int cutB = excess - cutA;
    if (magnitudeB > magnitudeA) std::swap(cutA, cutB);
    if (cutA > magnitudeA) { cutB += cutA - magnitudeA; cutA = magnitudeA; }
    if (cutB > magnitudeB) { cutA += cutB - magnitudeB; cutB = magnitudeB; }
    magnitudeA -= cutA;
    magnitudeB -= cutB;
    a = static_cast<std::int8_t>(a < 0 ? -magnitudeA : magnitudeA);
    b = static_cast<std::int8_t>(b < 0 ? -magnitudeB : magnitudeB);
}

} // namespace

void PackedMatrix::pack(const std::int8_t *source, const float *rowScales, int rowCount,
                        int rowWidth) {
    rows = rowCount;
    width = rowWidth;
    const int quantum = strideQuantum();
    stride = (rowWidth + quantum - 1) / quantum * quantum;
    notePacking(rows, width, stride);
    codes.assign(static_cast<std::size_t>(rows) * stride, 0);
    scales.assign(rowScales, rowScales + rows);
    sums.resize(static_cast<std::size_t>(rows));
    const bool constrain = pairConstraint();
    long long checked = 0;
    long long constrained = 0;
    long long cut = 0;
    for (int row = 0; row < rows; ++row) {
        std::int8_t *rowCodes = codes.data() + static_cast<std::size_t>(row) * stride;
        std::memcpy(rowCodes, source + static_cast<std::size_t>(row) * width,
                    static_cast<std::size_t>(width));
        if (constrain) {
            // Walk the pairs exactly as the u8 kernel multiplies them: adjacent elements, even index
            // first. The padding is zero, so a pair that straddles width is harmless either way.
            for (int at = 0; at + 1 < stride; at += 2) {
                const int a = rowCodes[at];
                const int b = rowCodes[at + 1];
                if ((a > 0) != (b > 0) || a == 0 || b == 0) continue; // opposite signs cannot saturate
                ++checked;
                if (std::abs(a) + std::abs(b) <= 128) continue;
                ++constrained;
                constrainPair(rowCodes[at], rowCodes[at + 1], cut);
            }
        }
        // From the CODES, not the source: the zero-point correction must describe the weights the
        // kernel will actually multiply, cuts included.
        std::int32_t sum = 0;
        for (int at = 0; at < width; ++at) {
            sum += rowCodes[at];
        }
        sums[static_cast<std::size_t>(row)] = sum;
    }
    if (constrain) {
        const std::lock_guard<std::mutex> guard(packingMutex);
        packingTotals.pairsChecked += checked;
        packingTotals.pairsConstrained += constrained;
        packingTotals.magnitudeCut += cut;
    }
}

namespace {

// Round to nearest even, matching the exporter's np.rint. On x86 this is one CVTSS2SI — lrintf
// is a library call on MinGW, and at ~10^5 quantised values per name it was MEASURABLE: several
// milliseconds of a ten-millisecond forward pass.
inline int roundToNearest(float value) noexcept {
#ifdef NEURELEASE_X86_64
    return _mm_cvt_ss2si(_mm_set_ss(value));
#else
    return static_cast<int>(std::lrintf(value));
#endif
}

inline float scaleFor(const float *values, int count, float limit) noexcept {
    float largest = 0.0F;
    int at = 0;
#ifdef NEURELEASE_X86_64
    const __m128 signMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128 best = _mm_setzero_ps();
    for (; at + 4 <= count; at += 4) {
        best = _mm_max_ps(best, _mm_and_ps(_mm_loadu_ps(values + at), signMask));
    }
    best = _mm_max_ps(best, _mm_shuffle_ps(best, best, _MM_SHUFFLE(1, 0, 3, 2)));
    best = _mm_max_ps(best, _mm_shuffle_ps(best, best, _MM_SHUFFLE(2, 3, 0, 1)));
    largest = _mm_cvtss_f32(best);
#endif
    for (; at < count; ++at) {
        largest = std::max(largest, std::fabs(values[at]));
    }
    return std::max(largest, 1e-12F) / limit;
}

} // namespace

float quantise(const float *values, int count, std::int8_t *destination,
               int paddedCount) noexcept {
    // The clamp is the mode's, not the caller's: the saturation-free u8 path needs +/-63 so its biased
    // activations stay inside [1, 127], and getting that wrong would reintroduce exactly the overflow
    // the mode exists to rule out.
    const int limit = activationLimit();
    const float scale = scaleFor(values, count, static_cast<float>(limit));
    const float inverse = 1.0F / scale;
    int at = 0;
#ifdef NEURELEASE_X86_64
    // CVTPS2DQ honours the current rounding mode (nearest even, same as the scalar path), and
    // PACKSSDW/PACKSSWB saturate, which the ±127.5 value bound makes identical to the clamp.
    // NO EXPLICIT CLAMP IN THE VECTOR LOOP, and none needed. The scale is largest/limit, so the
    // largest value in the block maps to exactly `limit` and everything else to less - the codes are
    // bounded by construction, not by clamping. Float rounding of the division can put one code a
    // single step over, and the narrow path has room for that: it stays safe up to a code of 65
    // (65 + 64 = 129 unsigned, and 129 * 127 * 2 = 32,766 against a 32,767 ceiling). PACKSSWB's own
    // saturation at +/-127 remains the backstop. The scalar tail below clamps properly because it is
    // free to.
    const __m128 factor = _mm_set1_ps(inverse);
    for (; at + 16 <= count; at += 16) {
        const __m128i a = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(values + at), factor));
        const __m128i b = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(values + at + 4), factor));
        const __m128i c = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(values + at + 8), factor));
        const __m128i d = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(values + at + 12), factor));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + at),
                         _mm_packs_epi16(_mm_packs_epi32(a, b), _mm_packs_epi32(c, d)));
    }
#endif
    for (; at < count; ++at) {
        const int rounded = roundToNearest(values[at] * inverse);
        destination[at] = static_cast<std::int8_t>(std::min(limit, std::max(-limit, rounded)));
    }
    std::memset(destination + count, 0, static_cast<std::size_t>(paddedCount - count));
    return scale;
}

float quantise16(const float *values, int count, std::int16_t *destination,
                 int paddedCount) noexcept {
    // Always the full range: int16 activations only ever feed the exact `vpmaddwd` paths.
    const float scale = scaleFor(values, count, 127.0F);
    const float inverse = 1.0F / scale;
    int at = 0;
#ifdef NEURELEASE_X86_64
    const __m128 factor = _mm_set1_ps(inverse);
    for (; at + 8 <= count; at += 8) {
        const __m128i low = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(values + at), factor));
        const __m128i high = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(values + at + 4), factor));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + at),
                         _mm_packs_epi32(low, high));
    }
#endif
    for (; at < count; ++at) {
        const int rounded = roundToNearest(values[at] * inverse);
        destination[at] = static_cast<std::int16_t>(std::min(127, std::max(-127, rounded)));
    }
    std::memset(destination + count, 0,
                static_cast<std::size_t>(paddedCount - count) * sizeof(std::int16_t));
    return scale;
}

ActivationKind activationKind(Path path) noexcept {
    return path == Path::Avx2 ? ActivationKind::Int16 : ActivationKind::Int8;
}

void matvec(const PackedMatrix &matrix, const void *activation,
            std::int32_t *accumulators) noexcept {
    resolveOnce();
    activeFunction.load(std::memory_order_relaxed)(matrix, activation, accumulators);
}

void matmul(const PackedMatrix &matrix, const void *activations, int actCount, int actStride,
            std::int32_t *accumulators) noexcept {
    resolveOnce();
    activeMatmul.load(std::memory_order_relaxed)(matrix, activations, actCount, actStride,
                                                 accumulators);
}

void finish(const PackedMatrix &matrix, const std::int32_t *accumulators, float activationScale,
            const float *bias, float *out) noexcept {
    for (int row = 0; row < matrix.rows; ++row) {
        const float base = bias != nullptr ? bias[row] : 0.0F;
        out[row] = base + static_cast<float>(accumulators[row]) * activationScale *
                              matrix.scales[static_cast<std::size_t>(row)];
    }
}

MatvecFunction scalarMatvec() noexcept { return &scalar; }

PathStatus pathStatus() noexcept {
    resolveOnce();
    return {activePathValue.load(std::memory_order_relaxed),
            demotedFlag.load(std::memory_order_relaxed),
            refusedPath.load(std::memory_order_relaxed)};
}

PackingStats packingStats() noexcept {
    const std::lock_guard<std::mutex> guard(packingMutex);
    return packingTotals;
}

void resetPackingStats() noexcept {
    const std::lock_guard<std::mutex> guard(packingMutex);
    packingTotals = PackingStats{};
}

void setUnsignedProbe(bool enabled, bool substitute) noexcept {
    probeSubstitute.store(substitute, std::memory_order_relaxed);
    probeEnabled.store(enabled, std::memory_order_relaxed);
}

UnsignedProbe unsignedProbe() noexcept {
    UnsignedProbe total;
    const std::lock_guard<std::mutex> guard(probeMutex);
    for (const ProbeCounters *counters : probeRegistry) {
        total.dotProducts += counters->dotProducts;
        total.pairs += counters->pairs;
        total.saturatedPairs += counters->saturatedPairs;
        total.dotProductsAffected += counters->dotProductsAffected;
        total.absoluteErrorSum += counters->absoluteErrorSum;
        if (counters->worstAbsoluteError > total.worstAbsoluteError)
            total.worstAbsoluteError = counters->worstAbsoluteError;
    }
    return total;
}

void resetUnsignedProbe() noexcept {
    const std::lock_guard<std::mutex> guard(probeMutex);
    for (ProbeCounters *counters : probeRegistry) *counters = ProbeCounters{};
}

void rememberBadPaths(const char *file) {
    const std::lock_guard<std::mutex> guard(resolveMutex);
    badPathFile = file == nullptr ? "" : file;
}

} // namespace neurelease::model::kernels
