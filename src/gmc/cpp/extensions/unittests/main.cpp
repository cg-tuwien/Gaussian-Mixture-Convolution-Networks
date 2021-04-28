#include <limits>
#include <chrono>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#ifdef NDEBUG
constexpr bool asserts_are_enabled = false;
#else
constexpr bool asserts_are_enabled = true;
#endif

TEST_CASE("check that asserts are enabled") {
    REQUIRE(asserts_are_enabled);
}


TEST_CASE("check that NaNs are enabled (-ffast-math removes support, -fno-finite-math-only puts it back in)") {
    REQUIRE(std::isnan(std::numeric_limits<float>::quiet_NaN() * float(std::chrono::system_clock::now().time_since_epoch().count())));
    REQUIRE(std::isnan(double(std::numeric_limits<float>::quiet_NaN() * float(std::chrono::system_clock::now().time_since_epoch().count()))));
}

TEST_CASE("check that we are also testing n_reduction 8 test cases (disable for faster building)") {
#ifdef GPE_LIMIT_N_REDUCTION
    constexpr bool testing_n_reduction_8_and_16 = false;
#else
    constexpr bool testing_n_reduction_8_and_16 = true;
#endif
    REQUIRE(testing_n_reduction_8_and_16);
}
