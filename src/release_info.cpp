#include "neurelease/release_info.hpp"

namespace neurelease {

std::string_view label(ResolutionTier value) noexcept {
    switch (value) {
    case ResolutionTier::P480: return "480p";
    case ResolutionTier::P720: return "720p";
    case ResolutionTier::P1080: return "1080p";
    case ResolutionTier::P1440: return "1440p";
    case ResolutionTier::P2160: return "2160p";
    case ResolutionTier::P4320: return "4320p";
    case ResolutionTier::Unknown: return {};
    }
    return {};
}

std::string_view label(SourceKind value) noexcept {
    switch (value) {
    case SourceKind::BluRay: return "BluRay";
    case SourceKind::WebDl: return "WEB-DL";
    case SourceKind::WebRip: return "WEBRip";
    case SourceKind::Web: return "WEB";
    case SourceKind::Hdtv: return "HDTV";
    case SourceKind::Dvd: return "DVD";
    case SourceKind::Cam: return "CAM";
    case SourceKind::Screener: return "Screener";
    case SourceKind::DigitalCinema: return "DCP";
    case SourceKind::Film: return "Film";
    case SourceKind::Unknown: return {};
    }
    return {};
}

std::string_view label(VideoCodec value) noexcept {
    switch (value) {
    case VideoCodec::Av1: return "AV1";
    case VideoCodec::Hevc: return "HEVC";
    case VideoCodec::H264: return "H.264";
    case VideoCodec::Xvid: return "XviD";
    case VideoCodec::Mpeg2: return "MPEG-2";
    case VideoCodec::Vp9: return "VP9";
    case VideoCodec::Vc1: return "VC-1";
    case VideoCodec::Wmv: return "WMV";
    case VideoCodec::Vvc: return "VVC";
    case VideoCodec::Vp8: return "VP8";
    case VideoCodec::RealVideo: return "RealVideo";
    case VideoCodec::Unknown: return {};
    }
    return {};
}

std::string_view label(MediumKind value) noexcept {
    switch (value) {
    case MediumKind::Video: return "video";
    case MediumKind::Music: return "music";
    case MediumKind::Audiobook: return "audiobook";
    case MediumKind::Book: return "book";
    case MediumKind::Comic: return "comic";
    case MediumKind::Software: return "software";
    case MediumKind::Game: return "game";
    case MediumKind::Image: return "image";
    case MediumKind::Subtitle: return "subtitle";
    case MediumKind::Archive: return "archive";
    case MediumKind::Unknown: return {};
    }
    return {};
}

std::string_view label(SubtitleFormat value) noexcept {
    switch (value) {
    case SubtitleFormat::Srt: return "SRT";
    case SubtitleFormat::Ass: return "ASS";
    case SubtitleFormat::VobSub: return "VOBSUB";
    case SubtitleFormat::Pgs: return "PGS";
    case SubtitleFormat::Vtt: return "VTT";
    case SubtitleFormat::Smi: return "SMI";
    case SubtitleFormat::Unknown: return {};
    }
    return {};
}

std::string_view label(EditionKind value) noexcept {
    switch (value) {
    case EditionKind::Imax: return "IMAX";
    case EditionKind::Criterion: return "Criterion";
    case EditionKind::OpenMatte: return "Open Matte";
    case EditionKind::Remastered: return "Remastered";
    case EditionKind::Unrated: return "Unrated";
    case EditionKind::Uncut: return "Uncut";
    case EditionKind::Uncensored: return "Uncensored";
    case EditionKind::SpecialEdition: return "Special Edition";
    case EditionKind::Deluxe: return "Deluxe";
    case EditionKind::Redux: return "Redux";
    case EditionKind::Extended: return "Extended";
    case EditionKind::DirectorsCut: return "Director's Cut";
    case EditionKind::FinalCut: return "Final Cut";
    case EditionKind::Internal: return "Internal";
    case EditionKind::Limited: return "Limited";
    case EditionKind::Untouched: return "Untouched";
    case EditionKind::Dirfix: return "Dirfix";
    case EditionKind::Custom: return "Custom";
    case EditionKind::Widescreen: return "Widescreen";
    case EditionKind::Download: return "Download";
    case EditionKind::Retail: return "Retail";
    case EditionKind::Collector: return "Collector";
    // "Final" and "Final Cut" are deliberately different strings: one marks the last episode of a
    // season, the other a recut of a film.
    case EditionKind::Final: return "Final";
    case EditionKind::Original: return "Original";
    case EditionKind::Fix: return "Fix";
    case EditionKind::CompleteEdition: return "Complete Edition";
    case EditionKind::Unabridged: return "Unabridged";
    case EditionKind::Reencode: return "Re-encode";
    case EditionKind::Numbered: return "Numbered Edition";
    case EditionKind::Regional: return "Regional";
    case EditionKind::HighQuality: return "High Quality";
    case EditionKind::Ultimate: return "Ultimate";
    case EditionKind::Censored: return "Censored";
    case EditionKind::FanEdit: return "Fan Edit";
    case EditionKind::Bootleg: return "Bootleg";
    case EditionKind::Unofficial: return "Unofficial";
    case EditionKind::Bonus: return "Bonus";
    case EditionKind::Festival: return "Festival";
    case EditionKind::MultiDisc: return "Multi-Disc";
    case EditionKind::AlternateCut: return "Alternate Cut";
    case EditionKind::Shortened: return "Shortened";
    case EditionKind::Leaked: return "Leaked";
    case EditionKind::Colorized: return "Colorized";
    case EditionKind::Fullscreen: return "Fullscreen";
    case EditionKind::Standard: return "Standard";
    case EditionKind::Creditless: return "Creditless";
    case EditionKind::ReRecorded: return "Re-recorded";
    case EditionKind::Commentary: return "Commentary";
    case EditionKind::Explicit: return "Explicit";
    case EditionKind::Reissue: return "Reissue";
    case EditionKind::OriginalAspectRatio: return "Original Aspect Ratio";
    case EditionKind::Restored: return "Restored";
    case EditionKind::Theatrical: return "Theatrical";
    case EditionKind::Despecialized: return "Despecialized";
    case EditionKind::AssemblyCut: return "Assembly Cut";
    case EditionKind::Anniversary: return "Anniversary";
    case EditionKind::Signature: return "Signature";
    case EditionKind::Imperial: return "Imperial";
    case EditionKind::Diamond: return "Diamond";
    case EditionKind::TwoInOne: return "2in1";
    case EditionKind::Preair: return "Preair";
    case EditionKind::Unknown: return {};
    }
    return {};
}

std::string_view fieldName(Field field) {
    switch (field) {
    case Field::Title:            return "title";
    case Field::Subtitle:         return "subtitle";
    case Field::AlternateTitle:   return "alternate_title";
    case Field::EpisodeTitle:     return "episode_title";
    case Field::Year:             return "year";
    case Field::AirDate:          return "air_date";
    case Field::Season:           return "season";
    case Field::Episode:          return "episode";
    case Field::AbsoluteEpisode:  return "absolute_episode";
    case Field::PackMarker:       return "pack_marker";
    case Field::SpecialMarker:    return "special_marker";
    case Field::Quality:          return "quality";
    case Field::ReleaseSource:    return "release_source";
    case Field::Platform:         return "platform";
    case Field::Edition:          return "edition";
    case Field::Codec:            return "codec";
    case Field::Hdr:              return "hdr";
    case Field::BitDepth:         return "bit_depth";
    case Field::Remux:            return "remux";
    case Field::Proper:           return "proper";
    case Field::Repack:           return "repack";
    case Field::AiUpscale:        return "ai_upscale";
    case Field::Hybrid:           return "hybrid";
    case Field::LightEncode:      return "light_encode";
    case Field::ThreeD:           return "three_d";
    case Field::Downscaled:       return "downscaled";
    case Field::Audio:            return "audio";
    case Field::AudioLanguage:    return "audio_language";
    case Field::SubtitleLanguage: return "subtitle_language";
    case Field::DualAudio:        return "dual_audio";
    case Field::MultiAudio:       return "multi_audio";
    case Field::MultiSubs:        return "multi_subs";
    case Field::HardSubs:         return "hard_subs";
    case Field::SubtitleFormat:   return "subtitle_format";
    case Field::EnglishDub:       return "english_dub";
    case Field::Group:            return "release_group";
    case Field::SiteBanner:       return "site_banner";
    case Field::TrackerTag:       return "tracker_tag";
    case Field::Container:        return "container";
    case Field::Medium:           return "medium";
    case Field::Crc32:            return "crc32";
    case Field::ContentKind:      return "content_kind";
    case Field::AdultKind:        return "adult_kind";
    case Field::NumberingKind:    return "numbering_kind";
    case Field::PackScope:        return "pack_scope";
    case Field::SpecialKind:      return "special_kind";
    case Field::FranchisePrefix:  return "franchise_prefix";
    }
    return {};
}

const FieldOrigin* originOf(const ReleaseInfo& info, Field field) {
    for (const FieldOrigin& origin : info.origins)
        if (origin.field == field) return &origin;
    return nullptr;
}

} // namespace neurelease
