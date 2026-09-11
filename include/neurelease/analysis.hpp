#pragma once

#include "neurelease/release_info.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace neurelease {

enum class SpanType : std::uint8_t {
    Unknown = 0,
    Main = 1, Alternate = 2, SubtitlePart = 3, FranchisePrefix = 4, EpisodeTitle = 5,
    PackMarker = 6, SpecialMarker = 7, Year = 8, Edition = 9, Resolution = 10,
    FrameSize = 11, SourceType = 12, Platform = 13, Codec = 14, BitDepth = 15, Hdr = 16,
    AudioCodec = 17, AudioFeature = 18, AudioChannels = 19, AudioLanguage = 20,
    SubtitleLanguage = 21, DualAudioTag = 22, HardsubTag = 23, ReleaseGroup = 24,
    SiteBanner = 25, TrackerTag = 26, SeasonMarker = 27, EpisodeMarker = 28,
    VolumeMarker = 29, Container = 30, FileExtension = 31, SizeTag = 32, Crc32 = 33,
    Other = 34,
    Outside = 35, // model background class "O"; decoded spans normally discard it
    // ONE SPAN THAT STATES TWO FIELDS. `Cap.104` is season 1 episode 4, and the pseudo-token
    // stream never breaks inside a run of digits, so no boundary can separate them. Appended
    // rather than placed beside EpisodeMarker because these values are public ABI and the
    // model resolves labels by name, which makes the number pure identity.
    SeasonEpisodeMarker = 36,
};

[[nodiscard]] std::string_view spanTypeName(SpanType type) noexcept;

struct SegmentedSpan {
    SpanType type = SpanType::Unknown;
    std::string typeLabel; // exported spelling, diagnostics only
    std::int32_t begin = 0;
    std::int32_t end = 0;
    float confidence = 0.0F;
};

struct SegmentedField {
    std::string value; // exported spelling, diagnostics only
    float confidence = 0.0F;
};

struct Analysis {
    bool valid = false;
    std::vector<SegmentedSpan> spans;

    SegmentedField contentKind, adultKind, numberingKind, packScope, specialKind, animeKind;
    ContentKind content = ContentKind::Unknown;
    AdultKind adult = AdultKind::Unknown;
    // Whether the release is anime, as the model read it. A model without the field leaves this
    // false and `animeKind.value` empty, which is how a caller tells "not anime" from "not asked".
    bool anime = false;
    NumberingKind numbering = NumberingKind::Unknown;
    PackScope pack = PackScope::Unknown;
    SpecialKind special = SpecialKind::Unknown;

    bool languageEvaluated = false;
    bool japanese = false;
    float japaneseProbability = 0.0F;

    std::string kernelPath;
    bool kernelDemoted = false;
    int tokens = 0;
    float encodeMicros = 0.0F;
    float modelMicros = 0.0F;
    float convolutionMicros = 0.0F;
    float matmulMicros = 0.0F;
    float attentionMicros = 0.0F;
};

} // namespace neurelease
