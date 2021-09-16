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

