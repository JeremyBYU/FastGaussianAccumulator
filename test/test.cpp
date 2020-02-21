#include "gaussian_integrator.hpp"
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"


TEST_CASE("test_test")
{
    const std::string value = gaussian_integrator::test();
    REQUIRE(value == std::string("test"));
}
