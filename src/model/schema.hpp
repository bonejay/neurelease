#pragma once

#include "neurelease/analysis.hpp"

#include <string_view>

namespace neurelease::model::schema {

[[nodiscard]] SpanType spanType(std::string_view label) noexcept;
[[nodiscard]] ContentKind contentKind(std::string_view label) noexcept;
[[nodiscard]] AdultKind adultKind(std::string_view label) noexcept;
[[nodiscard]] AnimeKind animeKind(std::string_view label) noexcept;
[[nodiscard]] NumberingKind numberingKind(std::string_view label) noexcept;
[[nodiscard]] PackScope packScope(std::string_view label) noexcept;
[[nodiscard]] SpecialKind specialKind(std::string_view label) noexcept;

} // namespace neurelease::model::schema
