#include "model/schema.hpp"

#include <array>
#include <utility>

namespace neurelease::model::schema {
namespace {

using namespace std::string_view_literals;

template <typename Enum, std::size_t Size>
Enum lookup(std::string_view label,
            const std::array<std::pair<std::string_view, Enum>, Size>& values) noexcept {
    for (const auto& [name, value] : values)
        if (name == label) return value;
    return Enum::Unknown;
}

} // namespace

SpanType spanType(std::string_view label) noexcept {
    static constexpr std::array values{
        std::pair{"MAIN"sv, SpanType::Main}, std::pair{"ALT"sv, SpanType::Alternate},
        std::pair{"subtitle_part"sv, SpanType::SubtitlePart},
        std::pair{"franchise_prefix"sv, SpanType::FranchisePrefix},
        std::pair{"episode_title"sv, SpanType::EpisodeTitle},
        std::pair{"pack_marker"sv, SpanType::PackMarker},
        std::pair{"special_marker"sv, SpanType::SpecialMarker},
        std::pair{"year"sv, SpanType::Year}, std::pair{"edition"sv, SpanType::Edition},
        std::pair{"resolution"sv, SpanType::Resolution},
        std::pair{"frame_size"sv, SpanType::FrameSize},
        std::pair{"source_type"sv, SpanType::SourceType},
        std::pair{"platform"sv, SpanType::Platform}, std::pair{"codec"sv, SpanType::Codec},
        std::pair{"bit_depth"sv, SpanType::BitDepth}, std::pair{"hdr"sv, SpanType::Hdr},
        std::pair{"audio_codec"sv, SpanType::AudioCodec},
        std::pair{"audio_feature"sv, SpanType::AudioFeature},
        std::pair{"audio_channels"sv, SpanType::AudioChannels},
        std::pair{"audio_language"sv, SpanType::AudioLanguage},
        std::pair{"subtitle_language"sv, SpanType::SubtitleLanguage},
        std::pair{"dual_audio_tag"sv, SpanType::DualAudioTag},
        std::pair{"hardsub_tag"sv, SpanType::HardsubTag},
        std::pair{"release_group"sv, SpanType::ReleaseGroup},
        std::pair{"site_banner"sv, SpanType::SiteBanner},
        std::pair{"tracker_tag"sv, SpanType::TrackerTag},
        std::pair{"season_marker"sv, SpanType::SeasonMarker},
        std::pair{"episode_marker"sv, SpanType::EpisodeMarker},
        std::pair{"season_episode_marker"sv, SpanType::SeasonEpisodeMarker},
        std::pair{"volume_marker"sv, SpanType::VolumeMarker},
        std::pair{"container"sv, SpanType::Container},
        std::pair{"file_extension"sv, SpanType::FileExtension},
        std::pair{"size_tag"sv, SpanType::SizeTag}, std::pair{"crc32"sv, SpanType::Crc32},
        std::pair{"other"sv, SpanType::Other}, std::pair{"O"sv, SpanType::Outside},
    };
    return lookup(label, values);
}

ContentKind contentKind(std::string_view label) noexcept {
    static constexpr std::array values{
        // THE VOCABULARY MODEL 3 SPEAKS. A name states the form and not the medium, so the four
        // live-action and animated values below are only ever seen from a model 2 file. Both
        // resolve, so either model loads and neither answer is silently renamed.
        std::pair{"movie"sv, ContentKind::Movie},
        std::pair{"series"sv, ContentKind::Series},
        std::pair{"live_action_movie"sv, ContentKind::LiveActionMovie},
        std::pair{"live_action_series"sv, ContentKind::LiveActionSeries},
        std::pair{"animated_movie"sv, ContentKind::AnimatedMovie},
        std::pair{"animated_series"sv, ContentKind::AnimatedSeries},
        std::pair{"music"sv, ContentKind::Music},
        std::pair{"book_document"sv, ContentKind::BookDocument},
        std::pair{"comic_manga"sv, ContentKind::ComicManga},
        std::pair{"software"sv, ContentKind::Software}, std::pair{"game"sv, ContentKind::Game},
        std::pair{"other"sv, ContentKind::Other},
    };
    return lookup(label, values);
}

AdultKind adultKind(std::string_view label) noexcept {
    static constexpr std::array values{
        std::pair{"false"sv, AdultKind::No}, std::pair{"true"sv, AdultKind::Yes},
        std::pair{"False"sv, AdultKind::No}, std::pair{"True"sv, AdultKind::Yes},
        std::pair{"0"sv, AdultKind::No}, std::pair{"1"sv, AdultKind::Yes},
    };
    return lookup(label, values);
}

// The sixth global field, and a boolean one: the model writes it the way Python prints it.
AnimeKind animeKind(std::string_view label) noexcept {
    static constexpr std::array values{
        std::pair{"False"sv, AnimeKind::No}, std::pair{"True"sv, AnimeKind::Yes},
        std::pair{"false"sv, AnimeKind::No}, std::pair{"true"sv, AnimeKind::Yes},
        std::pair{"0"sv, AnimeKind::No}, std::pair{"1"sv, AnimeKind::Yes},
    };
    return lookup(label, values);
}

NumberingKind numberingKind(std::string_view label) noexcept {
    static constexpr std::array values{
        std::pair{"season_episode"sv, NumberingKind::SeasonEpisode},
        std::pair{"absolute"sv, NumberingKind::Absolute},
        std::pair{"volume"sv, NumberingKind::Volume}, std::pair{"date"sv, NumberingKind::Date},
        std::pair{"none"sv, NumberingKind::None},
    };
    return lookup(label, values);
}

PackScope packScope(std::string_view label) noexcept {
    static constexpr std::array values{
        std::pair{"single"sv, PackScope::Single},
        std::pair{"episode_batch"sv, PackScope::EpisodeBatch},
        std::pair{"volume"sv, PackScope::Volume}, std::pair{"season"sv, PackScope::Season},
        std::pair{"multi_season"sv, PackScope::MultiSeason},
        std::pair{"complete"sv, PackScope::Complete},
    };
    return lookup(label, values);
}

SpecialKind specialKind(std::string_view label) noexcept {
    static constexpr std::array values{
        std::pair{"none"sv, SpecialKind::None}, std::pair{"ova"sv, SpecialKind::Ova},
        std::pair{"special"sv, SpecialKind::Special}, std::pair{"movie"sv, SpecialKind::Movie},
    };
    return lookup(label, values);
}

} // namespace neurelease::model::schema
