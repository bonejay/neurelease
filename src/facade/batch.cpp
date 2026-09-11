// The batch surface. Workers each hold a full Parser that shares the immutable model weights with
// the first one, names are grouped into similarly sized buckets so a batch of short names is not
// padded to its one long outlier, and results return in input order. Default worker count is half
// the logical cores; the constructor takes an explicit count and rp_set_batch_threads exposes the
// same choice through the C ABI.

#include "neurelease/parser.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace neurelease {
namespace {

BatchParser::Timing summarise(const std::vector<ParseResult>& results, int workers, int buckets,
                              double paddingPercent,
                              std::chrono::steady_clock::time_point started) {
    BatchParser::Timing timing;
    timing.names = results.size();
    timing.threads = workers;
    timing.buckets = buckets;
    timing.bucketPaddingPercent = paddingPercent;
    timing.wallMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    for (const ParseResult& result : results) {
        timing.encodeMilliseconds += result.analysis.encodeMicros / 1000.0;
        timing.modelMilliseconds += result.analysis.modelMicros / 1000.0;
        timing.convolutionMilliseconds += result.analysis.convolutionMicros / 1000.0;
        timing.matmulMilliseconds += result.analysis.matmulMicros / 1000.0;
        timing.attentionMilliseconds += result.analysis.attentionMicros / 1000.0;
    }
    return timing;
}

} // namespace

struct BatchParser::Implementation {
    std::vector<std::unique_ptr<Parser>> parsers;
    Timing timing;
    int bucketSize = 32;
};

BatchParser::BatchParser(std::string_view modelDirectory, int threadCount)
    : implementation_(std::make_unique<Implementation>()) {
    const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    const int wanted = threadCount > 0 ? threadCount : std::max(1, static_cast<int>(logical / 2));
    implementation_->parsers.reserve(static_cast<std::size_t>(wanted));
    // One worker loads and prepares the segmenter; the others share that immutable representation
    // while keeping their own encoder and scratch state.
    implementation_->parsers.push_back(std::make_unique<Parser>(modelDirectory));
    for (int index = 1; index < wanted; ++index) {
        implementation_->parsers.push_back(
            std::unique_ptr<Parser>(new Parser(modelDirectory, *implementation_->parsers.front())));
    }
}

BatchParser::~BatchParser() = default;
BatchParser::BatchParser(BatchParser&&) noexcept = default;
BatchParser& BatchParser::operator=(BatchParser&&) noexcept = default;

int BatchParser::threads() const noexcept {
    return static_cast<int>(implementation_->parsers.size());
}

void BatchParser::setBucketSize(int names) noexcept {
    implementation_->bucketSize = names > 0 ? names : 32;
}

int BatchParser::bucketSize() const noexcept { return implementation_->bucketSize; }

BatchParser::Timing BatchParser::lastTiming() const noexcept { return implementation_->timing; }

std::vector<ParseResult> BatchParser::parse(std::span<const std::string> names) {
    std::vector<ParseResult> results(names.size());
    implementation_->timing = {};
    if (names.empty()) return results;
    const auto started = std::chrono::steady_clock::now();
    const int workers = std::min(threads(), static_cast<int>(names.size()));
    if (workers <= 1) {
        for (std::size_t index = 0; index < names.size(); ++index)
            results[index] = implementation_->parsers.front()->parse(names[index]);
        implementation_->timing = summarise(results, 1, 1, 0.0, started);
        return results;
    }

    std::vector<std::uint32_t> order(names.size());
    for (std::size_t index = 0; index < order.size(); ++index)
        order[index] = static_cast<std::uint32_t>(index);
    std::stable_sort(order.begin(), order.end(), [names](std::uint32_t left, std::uint32_t right) {
        return names[left].size() > names[right].size();
    });

    const std::size_t bucket = static_cast<std::size_t>(std::max(1, implementation_->bucketSize));
    const std::size_t bucketCount = (order.size() + bucket - 1) / bucket;
    std::atomic<std::size_t> nextBucket{0};
    using Claimed = std::pair<std::uint32_t, ParseResult>;
    std::vector<std::vector<Claimed>> perWorker(static_cast<std::size_t>(workers));
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::mutex failureMutex;
    std::vector<std::jthread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        pool.emplace_back([this, worker, workers, names, bucket, bucketCount, &order, &nextBucket,
                           &perWorker, &failed, &failure, &failureMutex] {
            try {
                Parser& parser = *implementation_->parsers[static_cast<std::size_t>(worker)];
                std::vector<Claimed>& mine = perWorker[static_cast<std::size_t>(worker)];
                mine.reserve(names.size() / static_cast<std::size_t>(workers) + bucket);
                while (!failed.load(std::memory_order_relaxed)) {
                    const std::size_t claimed = nextBucket.fetch_add(1, std::memory_order_relaxed);
                    if (claimed >= bucketCount) break;
                    const std::size_t from = claimed * bucket;
                    const std::size_t end = std::min(from + bucket, order.size());
                    for (std::size_t at = from; at < end; ++at) {
                        if (failed.load(std::memory_order_relaxed)) break;
                        const std::uint32_t index = order[at];
                        mine.emplace_back(index, parser.parse(names[index]));
                    }
                }
            } catch (...) {
                std::scoped_lock lock(failureMutex);
                if (!failure) failure = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    pool.clear(); // jthread joins; results and the captured exception are now stable
    if (failure) std::rethrow_exception(failure);
    for (std::vector<Claimed>& mine : perWorker)
        for (Claimed& claimed : mine) results[claimed.first] = std::move(claimed.second);

    long long realLength = 0;
    long long paddedLength = 0;
    for (std::size_t from = 0; from < order.size(); from += bucket) {
        const std::size_t end = std::min(from + bucket, order.size());
        const long long longest = static_cast<long long>(names[order[from]].size());
        paddedLength += longest * static_cast<long long>(end - from);
        for (std::size_t at = from; at < end; ++at)
            realLength += static_cast<long long>(names[order[at]].size());
    }
    const double padding = paddedLength == 0 ? 0.0
        : 100.0 * static_cast<double>(paddedLength - realLength) / static_cast<double>(paddedLength);
    implementation_->timing = summarise(results, workers, static_cast<int>(bucketCount), padding, started);
    return results;
}

} // namespace neurelease
