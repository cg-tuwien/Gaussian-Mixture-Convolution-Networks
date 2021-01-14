#ifndef BVHCONFIG_H
#define BVHCONFIG_H
namespace lbvh {

struct Config {
    // 0 - 4 (inclusive, see morton_code.h)
    const enum class MortonCodeAlgorithm{Old, Cov1_12p36pc16i, Cov2_54pc10i, Cov3_27p27c10i, Cov4_27c27p10i} morton_code_algorithm = MortonCodeAlgorithm::Old;
};

}
#endif // BVHCONFIG_H
