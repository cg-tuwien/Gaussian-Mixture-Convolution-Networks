#include <catch2/catch.hpp>

#include "unittests/support.h"
#include "util/glm.h"
#include "util/grad/welford.h"
#include "util/welford.h"
#include "util/autodiff.h"

template<typename scalar_t, typename V>
void test_weighted_mean(const std::vector<scalar_t>& scalars, const std::vector<V>& values) {
    using autodiff_scalar = autodiff::Variable<scalar_t>;
    using autodiff_V = decltype (gpe::makeAutodiff(V()));

    std::vector<std::pair<scalar_t, V>> weight_value_pairs;
    for (const auto& w : scalars) {
        for (const auto& v : values) {
            weight_value_pairs.emplace_back(w, v);
        }
    }

    gpe::WeightedMean<scalar_t, V> aggregator;
    for (const auto& p : weight_value_pairs) {
        aggregator.addValue(p.first, p.second);
    }


    for (const auto& wsum_grad : scalars) {
        for (const auto& mean_grad : values) {
            std::vector<std::pair<autodiff_scalar, autodiff_V>> autodiff_weight_value_pairs;
            for (const auto& p : weight_value_pairs) {
                autodiff_weight_value_pairs.emplace_back(gpe::makeAutodiff(p.first), gpe::makeAutodiff(p.second));
            }

            gpe::WeightedMean<autodiff_scalar, autodiff_V> autodiff_aggregator;
            gpe::grad::WeightedMean<float, V> grad_aggregator(aggregator.w_sum, aggregator.mean(), wsum_grad, mean_grad);

            for (const auto& p : autodiff_weight_value_pairs) {
                autodiff_aggregator.addValue(p.first, p.second + autodiff_V{});
            }
            gpe::propagateGrad(autodiff_aggregator.w_sum, wsum_grad);
            gpe::propagateGrad(autodiff_aggregator.mean(), mean_grad);

            unsigned i = 0;
            for (const auto& p : weight_value_pairs) {
                scalar_t grad_weight = 0;
                V grad_value = {};
                grad_aggregator.addValue(p.first, p.second, &grad_weight, &grad_value);
                const auto reference_grad_weight = gpe::extractGrad(autodiff_weight_value_pairs[i].first);
                const auto reference_grad_value = gpe::extractGrad(autodiff_weight_value_pairs[i].second);
                REQUIRE(are_similar(grad_weight, reference_grad_weight));
                REQUIRE(are_similar(grad_value, reference_grad_value));
                i++;
            }
        }
    }
}

TEST_CASE("grad welford weighted mean") {
    SECTION( "scalars", "[grad_welford]" ) {
        test_weighted_mean(_smallPositiveScalarCollection(), _smallPositiveScalarCollection());
        test_weighted_mean(_smallNegativeScalarCollection(), _smallPositiveScalarCollection());
    }
    SECTION( "vectors", "[grad_welford]" ) {
        test_weighted_mean(_smallPositiveScalarCollection(), _smallVecCollection<2>());
        test_weighted_mean(_smallNegativeScalarCollection(), _smallVecCollection<2>());
//        test_weighted_mean(scalars, _vecCollection<3>());
    }
//    SECTION( "matrices", "[grad_welford]" ) {
//        const auto scalars = _scalarCollection();
//        test_weighted_mean(scalars, _vecCollection<2>());
//        test_weighted_mean(scalars, _covCollection<3>());
//    }
}
