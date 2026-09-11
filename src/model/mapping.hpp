#pragma once

#include "model/segmenter.hpp"
#include "neurelease/release_info.hpp"

#include <string_view>

namespace neurelease::model {

// Pure span-to-result mapping: no model load and no inference. `name` must be the exact UTF-8
// string the analysis addresses.
ReleaseInfo releaseInfoFromAnalysis(std::string_view name, const Analysis& analysis);

} // namespace neurelease::model
