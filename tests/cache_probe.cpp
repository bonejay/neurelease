// The cache head through the runtime: one name per line on stdin, one line out per name -
// the cached probability, or "-" when the model has no cache head. Not a test; the training
// repository's cache_prediction/score_cpp.py drives it to compare the int8 runtime with torch.
//
//   cache_probe <model directory> < names.txt > probabilities.txt

#include <cstdio>
#include <iostream>
#include <string>

#include "neurelease/parser.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cache_probe <model directory> < names\n");
        return 2;
    }
    std::ios::sync_with_stdio(false);
    neurelease::Parser parser(argv[1]);
    std::string name;
    while (std::getline(std::cin, name)) {
        if (!name.empty() && name.back() == '\r') {
            name.pop_back();
        }
        const auto result = parser.parse(name);
        if (result.analysis.cacheEvaluated) {
            std::printf("%.6f\n", static_cast<double>(result.analysis.cachedProbability));
        } else {
            std::printf("-\n");
        }
    }
    return 0;
}
