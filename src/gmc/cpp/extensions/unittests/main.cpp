#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#ifndef NDEBUG
constexpr bool asserts_are_enabled = true;
#else
constexpr bool asserts_are_enabled = false;
#endif

TEST_CASE("check that asserts are enabled") {
    REQUIRE(asserts_are_enabled);
}
