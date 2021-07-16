#include <catch2/catch.hpp>

#include "util/containers.h"
#include "util/helper.h"

using namespace gpe;

//todo: test for very large indices

TEMPLATE_TEST_CASE("util helpers", "[util]", int, unsigned, int64_t, uint64_t) {
    SECTION("indexing") {
        {
            const auto dims = Array<TestType, 4>{2, 3, 4, 5};
            REQUIRE(join_n_dim_index<TestType>(dims, {0, 0, 0, 0}) == 0);
            REQUIRE(join_n_dim_index<TestType>(dims, {1, 0, 0, 0}) == 1);
            REQUIRE(join_n_dim_index<TestType>(dims, {0, 1, 0, 0}) == 2);
            REQUIRE(join_n_dim_index<TestType>(dims, {1, 1, 0, 0}) == 3);
            REQUIRE(join_n_dim_index<TestType>(dims, {0, 0, 1, 0}) == 6);
            REQUIRE(join_n_dim_index<TestType>(dims, {1, 1, 1, 0}) == 9);
            REQUIRE(join_n_dim_index<TestType>(dims, {0, 0, 0, 1}) == 24);
            REQUIRE(join_n_dim_index<TestType>(dims, {1, 1, 1, 1}) == 33);

            for (TestType i = 0; i < 2; ++i) {
                for (TestType j = 0; j < 3; ++j) {
                    for (TestType k = 0; k < 4; ++k) {
                        for (TestType l = 0; l < 5; ++l) {
                            const auto joined = join_n_dim_index<TestType>(dims, {i, j, k, l});
                            const auto split = split_n_dim_index(dims, joined);
                            REQUIRE(split[0] == i);
                            REQUIRE(split[1] == j);
                            REQUIRE(split[2] == k);
                            REQUIRE(split[3] == l);
                        }
                    }
                }
            }
        }
        {
            const auto dims = Array<TestType, 3>{(1<<17), (1<<12), (1<<4)};
            REQUIRE(join_n_dim_index<uint64_t>(dims, {0, 0, 0}) == 0);
            REQUIRE(join_n_dim_index<uint64_t>(dims, {1, 0, 0}) == 1);
            REQUIRE(join_n_dim_index<uint64_t>(dims, {0, 1, 0}) == (1<<17));
            REQUIRE(join_n_dim_index<uint64_t>(dims, {0, 0, 1}) == (1<<17) * (1<<12));
            {
                for (TestType i = 0; i < 2; ++i) {
                    for (TestType j = 0; j < 3; ++j) {
                        for (TestType k = 0; k < 4; ++k) {
                            const auto joined = join_n_dim_index<uint64_t>(dims, {i, j, k});
                            const auto split = split_n_dim_index(dims, joined);
                            REQUIRE(split[0] == i);
                            REQUIRE(split[1] == j);
                            REQUIRE(split[2] == k);
                        }
                    }
                }
            }

        }
    }
}
