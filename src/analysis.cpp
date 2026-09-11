#include "neurelease/analysis.hpp"

namespace neurelease {

std::string_view spanTypeName(SpanType type) noexcept {
    switch (type) {
    case SpanType::Unknown: return "unknown";
    case SpanType::Main: return "MAIN";
    case SpanType::Alternate: return "ALT";
    case SpanType::SubtitlePart: return "subtitle_part";
    case SpanType::FranchisePrefix: return "franchise_prefix";
    case SpanType::EpisodeTitle: return "episode_title";
    case SpanType::SeasonEpisodeMarker: return "season_episode_marker";
    case SpanType::PackMarker: return "pack_marker";
    case SpanType::SpecialMarker: return "special_marker";
    case SpanType::Year: return "year";
    case SpanType::Edition: return "edition";
    case SpanType::Resolution: return "resolution";
    case SpanType::FrameSize: return "frame_size";
    case SpanType::SourceType: return "source_type";
    case SpanType::Platform: return "platform";
    case SpanType::Codec: return "codec";
    case SpanType::BitDepth: return "bit_depth";
    case SpanType::Hdr: return "hdr";
    case SpanType::AudioCodec: return "audio_codec";
    case SpanType::AudioFeature: return "audio_feature";
    case SpanType::AudioChannels: return "audio_channels";
    case SpanType::AudioLanguage: return "audio_language";
    case SpanType::SubtitleLanguage: return "subtitle_language";
    case SpanType::DualAudioTag: return "dual_audio_tag";
    case SpanType::HardsubTag: return "hardsub_tag";
    case SpanType::ReleaseGroup: return "release_group";
    case SpanType::SiteBanner: return "site_banner";
    case SpanType::TrackerTag: return "tracker_tag";
    case SpanType::SeasonMarker: return "season_marker";
    case SpanType::EpisodeMarker: return "episode_marker";
    case SpanType::VolumeMarker: return "volume_marker";
    case SpanType::Container: return "container";
    case SpanType::FileExtension: return "file_extension";
    case SpanType::SizeTag: return "size_tag";
    case SpanType::Crc32: return "crc32";
    case SpanType::Other: return "other";
    case SpanType::Outside: return "O";
    }
    return "unknown";
}

} // namespace neurelease
