#include <neurelease/parser.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> namesFrom(const char* path) {
    if (path == nullptr) {
        return {
            "Game.of.Thrones.S01E08.1080p.WEB-DL.x264-GRP.mkv",
            "Blade.Runner.2049.2017.2160p.UHD.BluRay.x265-TERMiNAL.mkv",
            "The.Wire.S01-S05.COMPLETE.1080p.BluRay.x265-GRP",
            "Movie.2024.2160p.AMZN.WEB-DL.HEVC.10bit.DV.DDP.5.1.Atmos-GROUP.mkv",
        };
    }
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("cannot open names file: ") + path);
    std::vector<std::string> names;
    for (std::string line; std::getline(input, line);)
        if (!line.empty()) names.push_back(std::move(line));
    if (names.empty()) throw std::runtime_error("names file is empty");
    return names;
}

template <typename Run>
double timeMicrosPerName(std::size_t names, int repetitions, Run run) {
    const auto begin = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition) run();
    const auto elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin).count();
    return elapsed / static_cast<double>(names * static_cast<std::size_t>(repetitions));
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: parse_benchmark MODEL_DIR [NAMES_FILE] [REPETITIONS] [WORKERS=4]\n";
        return 2;
    }
    const std::vector<std::string> names = namesFrom(argc >= 3 ? argv[2] : nullptr);
    const int repetitions = argc >= 4 ? std::max(1, std::stoi(argv[3])) : 20;
    // Four workers by default: the number a typical deployment grants a background scan, and low
    // enough that the figure is not flattered by a big development machine. Argument 4 overrides.
    const int workers = argc >= 5 ? std::max(1, std::stoi(argv[4])) : 4;

    neurelease::Parser single(argv[1]);
    neurelease::BatchParser batch(argv[1], workers);
    for (const std::string& name : names) (void)single.parse(name);
    (void)batch.parse(names);

    // Batch first: the single-name loop is half a minute of full single-core load, and measuring
    // the batch after it means measuring a thermally throttled machine against a fresh one.
    const double batchMicros = timeMicrosPerName(names.size(), repetitions, [&] {
        (void)batch.parse(names);
    });
    const double singleMicros = timeMicrosPerName(names.size(), repetitions, [&] {
        for (const std::string& name : names) (void)single.parse(name);
    });
    const auto timing = batch.lastTiming();

    std::cout << std::fixed << std::setprecision(2)
              << "names=" << names.size() << " repetitions=" << repetitions
              << " workers=" << batch.threads() << '\n'
              << "single_us_per_name=" << singleMicros << '\n'
              << "batch_us_per_name=" << batchMicros << '\n'
              << "speedup=" << singleMicros / batchMicros << "x\n"
              << "last_batch_padding_percent=" << timing.bucketPaddingPercent << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
