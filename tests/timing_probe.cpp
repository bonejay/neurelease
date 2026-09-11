#include "neurelease/parser.hpp"
#include "model/kernels/kernels.hpp"

static const char* probePathName(neurelease::model::kernels::Path p) {
    switch (p) {
    case neurelease::model::kernels::Path::Scalar: return "Scalar";
    case neurelease::model::kernels::Path::Avx2: return "Avx2";
    case neurelease::model::kernels::Path::Avx2Vnni: return "Avx2Vnni";
    case neurelease::model::kernels::Path::Avx512Vnni: return "Avx512Vnni";
    }
    return "?";
}
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    std::string models = argv[1];
    std::vector<std::string> names;
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);) if (!line.empty()) names.push_back(line);
    neurelease::BatchParser batch(models, argc > 3 ? std::atoi(argv[3]) : 0);
    batch.parse(names);                       // warm
    auto parsed = batch.parse(names);
    auto t = batch.lastTiming();
    std::cout << "simd path        " << probePathName(neurelease::model::kernels::activePath()) << std::endl;
    std::cout << "names " << t.names << " threads " << t.threads << " buckets " << t.buckets << "\n"
              << "wall/name        " << t.averageWallMicrosPerName() << " us\n"
              << "encode  total    " << t.encodeMilliseconds << " ms\n"
              << "model   total    " << t.modelMilliseconds << " ms\n"
              << "  convolution    " << t.convolutionMilliseconds << " ms\n"
              << "  matmul         " << t.matmulMilliseconds << " ms\n"
              << "  attention      " << t.attentionMilliseconds << " ms\n"
              << "padding          " << t.bucketPaddingPercent << " %\n";
    return 0;
}
