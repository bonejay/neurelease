#include <neurelease/release_parser.h>

_Static_assert(RP_ABI_VERSION == 4u, "unexpected ABI version");
_Static_assert(RP_SPECIAL_MOVIE == 4, "enum values are part of the ABI");
_Static_assert(RP_EDITION_PREAIR == 22, "enum values are part of the ABI");
_Static_assert(RP_RESOLUTION_2160P == 5, "enum values are part of the ABI");
_Static_assert(RP_CODEC_HEVC == 2, "enum values are part of the ABI");

int main(void) {
    rp_origin_view origin = {0};
    rp_date date = {0};
    return origin.begin != 0 || date.year != 0;
}
