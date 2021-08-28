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
            gpe::grad::WeightedMean<scalar_t, V> grad_aggregator(aggregator.w_sum, aggregator.mean(), wsum_grad, mean_grad);

            for (const auto& p : autodiff_weight_value_pairs) {
                autodiff_aggregator.addValue(p.first, p.second);
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

template<typename scalar_t, int N>
void test_weighted_mean_and_cov(const std::vector<scalar_t>& scalars, const std::vector<glm::vec<N, scalar_t>>& values) {
    using vec_t = glm::vec<N, scalar_t>;
    using autodiff_scalar = autodiff::Variable<scalar_t>;
    using autodiff_V = decltype (gpe::makeAutodiff(values.front()));

    std::vector<std::pair<scalar_t, vec_t>> weight_value_pairs;
    for (const auto& w : scalars) {
        for (const auto& v : values) {
            weight_value_pairs.emplace_back(w, v);
        }
    }

    gpe::WeightedMeanAndCov<N, scalar_t> aggregator;
    for (const auto& p : weight_value_pairs) {
        aggregator.addValue(p.first, p.second);
    }


    for (const auto& wsum_grad : scalars) {
        for (const auto& mean_grad : values) {
            for (const auto& cov_grad : _covCollection<N, scalar_t>()) {
                std::vector<std::pair<autodiff_scalar, autodiff_V>> autodiff_weight_value_pairs;
                for (const auto& p : weight_value_pairs) {
                    autodiff_weight_value_pairs.emplace_back(gpe::makeAutodiff(p.first), gpe::makeAutodiff(p.second));
                }

                gpe::WeightedMeanAndCov<N, autodiff_scalar> autodiff_aggregator;
                gpe::grad::WeightedMeanAndCov<N, scalar_t> grad_aggregator(aggregator.w_sum, aggregator.mean(), aggregator.cov_matrix(), wsum_grad, mean_grad, cov_grad);

                for (const auto& p : autodiff_weight_value_pairs) {
                    autodiff_aggregator.addValue(p.first, p.second);
                }
                gpe::propagateGrad(autodiff_aggregator.w_sum, wsum_grad);
                gpe::propagateGrad(autodiff_aggregator.mean(), mean_grad);
                gpe::propagateGrad(autodiff_aggregator.cov_matrix(), cov_grad);

                unsigned i = 0;
                for (const auto& p : weight_value_pairs) {
                    scalar_t grad_weight = 0;
                    vec_t grad_value = {};
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
}

TEST_CASE("grad welford weighted mean") {
    SECTION( "scalars", "[grad_welford]" ) {
        test_weighted_mean(_smallPositiveScalarCollection<double>(), _smallPositiveScalarCollection<double>());
        test_weighted_mean(_smallNegativeScalarCollection<double>(), _smallPositiveScalarCollection<double>());
    }
    SECTION( "vectors", "[grad_welford]" ) {
        test_weighted_mean(_smallPositiveScalarCollection<double>(), _smallVecCollection<2, double>());
        test_weighted_mean(_smallNegativeScalarCollection<double>(), _smallVecCollection<2, double>());
        test_weighted_mean(_smallPositiveScalarCollection<double>(), _smallVecCollection<3, double>());
        test_weighted_mean(_smallNegativeScalarCollection<double>(), _smallVecCollection<3, double>());
    }
    SECTION( "matrices", "[grad_welford]" ) {
        test_weighted_mean(_smallPositiveScalarCollection<double>(), _smallCovCollection<2, double>());
        test_weighted_mean(_smallPositiveScalarCollection<double>(), _smallCovCollection<3, double>());
        test_weighted_mean(_smallNegativeScalarCollection<double>(), _smallCovCollection<2, double>());
        test_weighted_mean(_smallNegativeScalarCollection<double>(), _smallCovCollection<3, double>());
    }
}

TEST_CASE("grad welford weighted mean and cov") {
    SECTION( "vectors", "[grad_welford]" ) {
        test_weighted_mean_and_cov(_smallPositiveScalarCollection<double>(), _smallVecCollection<2, double>());
        test_weighted_mean_and_cov(_smallNegativeScalarCollection<double>(), _smallVecCollection<2, double>());
        test_weighted_mean_and_cov(_smallPositiveScalarCollection<double>(), _smallVecCollection<3, double>());
        test_weighted_mean_and_cov(_smallNegativeScalarCollection<double>(), _smallVecCollection<3, double>());
    }
}

