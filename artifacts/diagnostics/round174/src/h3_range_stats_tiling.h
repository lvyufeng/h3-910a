#ifndef H3_RANGE_STATS_TILING_H
#define H3_RANGE_STATS_TILING_H
// Plain C types on purpose: the AI-core compile stage has no C++ standard library,
// so pulling in <cstdint> here breaks the device build.
struct H3RangeStatsTilingData {
    unsigned int size;
    unsigned int coreNum;
};
#endif
