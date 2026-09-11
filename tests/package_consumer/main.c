#include <neurelease/release_parser.h>

#include <stdio.h>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    rp_parser* parser = NULL;
    rp_result* result = NULL;
    if (rp_parser_new(argv[1], &parser) != RP_OK) {
        fprintf(stderr, "%s\n", rp_global_error_message());
        return 3;
    }
    const rp_status status = rp_parse(parser, "Movie.2024.1080p.WEB-DL.HEVC-GRP.mkv", &result);
    const int valid = status == RP_OK && rp_flag(result, RP_FLAG_VALID) &&
                      rp_screen_size(result) == RP_RESOLUTION_1080P;
    rp_result_free(result);
    rp_parser_free(parser);
    return valid ? 0 : 4;
}
