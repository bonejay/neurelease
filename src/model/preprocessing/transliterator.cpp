#include "model/preprocessing/transliterator.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace neurelease::model::preprocessing {
namespace {

constexpr auto Missing = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t MaximumMatchCodepoints = 12;

template <typename T> T read(std::ifstream &input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated contextual transliteration model");
    }
    return value;
}

} // namespace

ContextualTransliterator::ContextualTransliterator(const std::filesystem::path &modelPath) {
    std::ifstream input(modelPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open contextual transliteration model " +
                                 modelPath.string());
    }

    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (std::memcmp(magic, "RCTXTR1\0", sizeof(magic)) != 0) {
        throw std::runtime_error("invalid contextual transliteration model magic");
    }
    if (read<std::uint32_t>(input) != 1) {
        throw std::runtime_error("unsupported contextual transliteration model version");
    }

    const auto nodeCount = read<std::uint32_t>(input);
    const auto edgeCount = read<std::uint32_t>(input);
    const auto terminalCount = read<std::uint32_t>(input);
    const auto bagCount = read<std::uint32_t>(input);
    const auto letterCount = read<std::uint32_t>(input);

    nodes_.reserve(nodeCount);
    for (std::uint32_t index = 0; index < nodeCount; ++index) {
        const auto edgeOffset = read<std::uint32_t>(input);
        const auto count = read<std::uint16_t>(input);
        const auto terminal = read<std::uint32_t>(input);
        nodes_.push_back({edgeOffset, terminal, count});
    }

    edges_.reserve(edgeCount);
    for (std::uint32_t index = 0; index < edgeCount; ++index) {
        edges_.push_back({read<std::uint32_t>(input), read<std::uint32_t>(input)});
    }

    std::size_t terminalBagCount = 0;
    terminals_.reserve(terminalCount);
    for (std::uint32_t index = 0; index < terminalCount; ++index) {
        const auto bagOffset = read<std::uint32_t>(input);
        const auto count = read<std::uint16_t>(input);
        terminals_.push_back({bagOffset, count});
        terminalBagCount += count;
    }

    terminalBags_.resize(terminalBagCount);
    input.read(reinterpret_cast<char *>(terminalBags_.data()),
               static_cast<std::streamsize>(terminalBagCount * sizeof(std::uint32_t)));

    bags_.reserve(bagCount);
    for (std::uint32_t index = 0; index < bagCount; ++index) {
        bags_.push_back({read<std::uint32_t>(input), read<std::uint16_t>(input)});
    }

    letters_.resize(letterCount);
    input.read(reinterpret_cast<char *>(letters_.data()), letterCount);
    if (!input || nodes_.empty()) {
        throw std::runtime_error("invalid contextual transliteration model payload");
    }

    for (const auto &node : nodes_) {
        if (node.edgeOffset + node.edgeCount > edges_.size() ||
            (node.terminal != Missing && node.terminal >= terminals_.size())) {
            throw std::runtime_error("contextual transliteration node is out of bounds");
        }
    }
    for (const auto &edge : edges_) {
        if (edge.child >= nodes_.size()) {
            throw std::runtime_error("contextual transliteration edge is out of bounds");
        }
    }
    for (const auto &terminal : terminals_) {
        if (terminal.bagOffset + terminal.bagCount > terminalBags_.size()) {
            throw std::runtime_error("contextual transliteration terminal is out of bounds");
        }
    }
    for (const auto bagIndex : terminalBags_) {
        if (bagIndex >= bags_.size()) {
            throw std::runtime_error("contextual transliteration bag reference is out of bounds");
        }
    }
    for (const auto &bag : bags_) {
        if (bag.letterCount > ContextualPosition{}.letters.size() ||
            bag.letterOffset + bag.letterCount > letters_.size()) {
            throw std::runtime_error("contextual transliteration letter bag is out of bounds");
        }
    }
    if (std::any_of(
            letters_.begin(), letters_.end(), [](std::uint8_t letter) { return letter >= 26; })) {
        throw std::runtime_error("contextual transliteration contains a non-Latin letter ID");
    }

    modelFileBytes_ = std::filesystem::file_size(modelPath);
}

void ContextualTransliterator::encode(std::span<const std::uint32_t> codepoints,
                                      std::vector<ContextualPosition> &output) const {
    output.assign(codepoints.size(), {});

    for (std::size_t start = 0; start < codepoints.size();) {
        auto node = std::uint32_t{0};
        auto terminal = Missing;
        auto matchedEnd = start;
        const auto limit = std::min(codepoints.size(), start + MaximumMatchCodepoints);

        for (auto position = start; position < limit; ++position) {
            node = child(node, codepoints[position]);
            if (node == Missing) {
                break;
            }
            if (nodes_[node].terminal != Missing) {
                terminal = nodes_[node].terminal;
                matchedEnd = position + 1;
            }
        }

        if (terminal == Missing) {
            ++start;
            continue;
        }

        copyMatch(terminal, start, matchedEnd - start, output);
        start = matchedEnd;
    }
}

std::uint32_t ContextualTransliterator::child(std::uint32_t node,
                                              std::uint32_t codepoint) const noexcept {
    const auto &value = nodes_[node];
    const auto first = edges_.begin() + value.edgeOffset;
    const auto last = first + value.edgeCount;
    const auto found =
        std::lower_bound(first, last, codepoint, [](const Edge &edge, std::uint32_t sought) {
            return edge.codepoint < sought;
        });
    return found != last && found->codepoint == codepoint ? found->child : Missing;
}

void ContextualTransliterator::copyMatch(std::uint32_t terminalIndex,
                                         std::size_t sourceStart,
                                         std::size_t sourceLength,
                                         std::vector<ContextualPosition> &output) const {
    const auto &terminal = terminals_[terminalIndex];
    if (terminal.bagCount != sourceLength) {
        throw std::runtime_error("contextual terminal does not preserve source positions");
    }

    for (std::size_t offset = 0; offset < sourceLength; ++offset) {
        const auto bagIndex = terminalBags_[terminal.bagOffset + offset];
        const auto &bag = bags_[bagIndex];
        auto &position = output[sourceStart + offset];
        position.letterCount = static_cast<std::uint8_t>(bag.letterCount);
        position.matchOffset = static_cast<std::uint8_t>(offset);
        position.matchLength = static_cast<std::uint8_t>(sourceLength);
        std::copy_n(letters_.begin() + bag.letterOffset, bag.letterCount, position.letters.begin());
    }
}

std::uintmax_t ContextualTransliterator::modelFileBytes() const noexcept {
    return modelFileBytes_;
}

std::size_t ContextualTransliterator::runtimeTableBytes() const noexcept {
    return nodes_.capacity() * sizeof(Node) + edges_.capacity() * sizeof(Edge) +
           terminals_.capacity() * sizeof(Terminal) +
           terminalBags_.capacity() * sizeof(std::uint32_t) + bags_.capacity() * sizeof(Bag) +
           letters_.capacity();
}

} // namespace neurelease::model::preprocessing
