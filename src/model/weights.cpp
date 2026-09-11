#include "model/weights.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace neurelease::model {

namespace {

constexpr char Magic[8]{'R', 'P', 'S', 'E', 'G', '2', '\0', '\0'};
constexpr std::uint32_t SupportedVersion = 3;

// A cursor over the loaded bytes that refuses to read past the end. Every read in the parser goes
// through it, so a truncated file becomes one exception rather than undefined behaviour.
class Reader {
    public:
    Reader(const std::vector<char> &bytes) : bytes_(bytes) {}

    template <typename Value> Value take() {
        Value value;
        copy(&value, sizeof(Value));
        return value;
    }

    std::string takeString(std::size_t length) {
        std::string value(length, '\0');
        copy(value.data(), length);
        return value;
    }

    void copyFrom(std::size_t offset, void *destination, std::size_t length) const {
        if (offset > bytes_.size() || length > bytes_.size() - offset) {
            throw std::runtime_error("segmenter weights: payload runs past the end of the file");
        }
        std::memcpy(destination, bytes_.data() + offset, length);
    }

    private:
    void copy(void *destination, std::size_t length) {
        copyFrom(at_, destination, length);
        at_ += length;
    }

    const std::vector<char> &bytes_;
    std::size_t at_ = 0;
};

std::vector<char> readWholeFile(const std::string &path) {
    // The string is UTF-8 by contract. On Windows a char* constructor would read it as the ANSI
    // codepage and a model directory with an umlaut or kanji in it silently stops opening, so the
    // bytes go through a u8 filesystem::path, which decodes UTF-8 on every platform.
    const std::filesystem::path decoded{std::u8string(path.begin(), path.end())};
    std::ifstream stream(decoded, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("segmenter weights: cannot open " + path);
    }
    const auto size = stream.tellg();
    std::vector<char> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    stream.read(bytes.data(), size);
    if (!stream) {
        throw std::runtime_error("segmenter weights: short read from " + path);
    }
    return bytes;
}

} // namespace

std::int64_t Tensor::count() const noexcept {
    std::int64_t product = 1;
    for (const auto dimension : shape) {
        product *= dimension;
    }
    return shape.empty() ? 0 : product;
}

std::int64_t Tensor::rowWidth() const noexcept {
    return rows() > 0 ? count() / rows() : 0;
}

struct Weights::Storage {
    std::map<std::string, Tensor, std::less<>> tensors;
    std::vector<std::string> names;
};

Weights Weights::load(const std::string &path) {
    const auto bytes = readWholeFile(path);
    Reader reader(bytes);

    if (bytes.size() < sizeof(Magic) || std::memcmp(bytes.data(), Magic, sizeof(Magic)) != 0) {
        throw std::runtime_error("segmenter weights: " + path + " is not an RPSEG2 artifact");
    }
    reader.takeString(sizeof(Magic));
    const auto version = reader.take<std::uint32_t>();
    if (version != SupportedVersion) {
        throw std::runtime_error(
            "segmenter weights: version " + std::to_string(version) + " in " + path +
            ", this reader needs " + std::to_string(SupportedVersion) +
            "; re-export with training/export_segmenter_weights.py");
    }
    const auto count = reader.take<std::uint32_t>();

    struct Entry {
        std::string name;
        std::vector<std::int64_t> shape;
        std::uint8_t dtype = 0;
        std::uint32_t offset = 0, bytes = 0, scaleOffset = 0, scaleBytes = 0;
    };
    std::vector<Entry> entries(count);
    for (auto &entry : entries) {
        entry.name = reader.takeString(reader.take<std::uint16_t>());
        const auto dimensions = reader.take<std::uint8_t>();
        entry.shape.reserve(dimensions);
        for (std::uint8_t index = 0; index < dimensions; ++index) {
            entry.shape.push_back(reader.take<std::uint32_t>());
        }
        entry.dtype = reader.take<std::uint8_t>();
        entry.offset = reader.take<std::uint32_t>();
        entry.bytes = reader.take<std::uint32_t>();
        entry.scaleOffset = reader.take<std::uint32_t>();
        entry.scaleBytes = reader.take<std::uint32_t>();
    }
    const auto payload = reader.take<std::uint32_t>();

    auto storage = std::make_shared<Storage>();
    for (const auto &entry : entries) {
        Tensor tensor;
        tensor.shape = entry.shape;
        const auto count64 = tensor.count();

        switch (entry.dtype) {
        case 0: // float32
            if (entry.bytes != count64 * sizeof(float)) {
                throw std::runtime_error("segmenter weights: " + entry.name +
                                         " float payload does not match its shape");
            }
            tensor.floats.resize(static_cast<std::size_t>(count64));
            reader.copyFrom(payload + entry.offset, tensor.floats.data(), entry.bytes);
            break;
        case 1: // int8 codes, one float scale per row
            if (entry.bytes != count64 ||
                entry.scaleBytes != tensor.rows() * sizeof(float)) {
                throw std::runtime_error("segmenter weights: " + entry.name +
                                         " int8 payload does not match its shape");
            }
            tensor.codes.resize(static_cast<std::size_t>(count64));
            reader.copyFrom(payload + entry.offset, tensor.codes.data(), entry.bytes);
            tensor.scales.resize(static_cast<std::size_t>(tensor.rows()));
            reader.copyFrom(payload + entry.scaleOffset, tensor.scales.data(), entry.scaleBytes);
            break;
        case 2: // UTF-8 text
            tensor.text.resize(entry.bytes);
            reader.copyFrom(payload + entry.offset, tensor.text.data(), entry.bytes);
            break;
        default:
            throw std::runtime_error("segmenter weights: " + entry.name + " has unknown dtype " +
                                     std::to_string(entry.dtype));
        }
        storage->names.push_back(entry.name);
        if (!storage->tensors.emplace(entry.name, std::move(tensor)).second) {
            throw std::runtime_error("segmenter weights: duplicate tensor " + entry.name);
        }
    }

    Weights weights;
    weights.storage_ = std::move(storage);
    return weights;
}

bool Weights::has(std::string_view name) const noexcept {
    return storage_ && storage_->tensors.find(name) != storage_->tensors.end();
}

const Tensor &Weights::tensor(std::string_view name) const {
    if (storage_) {
        const auto found = storage_->tensors.find(name);
        if (found != storage_->tensors.end()) {
            return found->second;
        }
    }
    throw std::runtime_error("segmenter weights: tensor " + std::string(name) + " is missing");
}

const std::string &Weights::text(std::string_view name) const { return tensor(name).text; }

float Weights::scalar(std::string_view name) const {
    const auto &found = tensor(name);
    if (found.floats.empty()) {
        throw std::runtime_error("segmenter weights: " + std::string(name) + " is not a scalar");
    }
    return found.floats.front();
}

const std::vector<std::string> &Weights::names() const noexcept { return storage_->names; }

} // namespace neurelease::model
