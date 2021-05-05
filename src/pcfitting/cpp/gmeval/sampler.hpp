/*
From https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
*/

#include <Eigen/Eigen>
#include <random>

struct normal_random_variable
{
    normal_random_variable() {}

    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

torch::Tensor sample_gmm(torch::Tensor gmm, int count)
{
    auto gmmAccess = gmm.accessor<float, 2>();

    int N = gmmAccess.size(0);
    std::vector<float> accweights(N, 0);
    accweights[0] = gmmAccess[0][0];
    for (int i = 1; i < N; ++i)
    {
        accweights[i] = accweights[i-1] + gmmAccess[i][0];
    }
    std::vector<normal_random_variable> gaussians(N);
    for (int i = 0; i < N; ++i)
    {
        Eigen::Matrix3f covar;
        covar << (float)gmmAccess[i][4], (float)gmmAccess[i][5], (float)gmmAccess[i][6],
            (float)gmmAccess[i][7], (float)gmmAccess[i][8], (float)gmmAccess[i][9], 
            (float)gmmAccess[i][10], (float)gmmAccess[i][11], (float)gmmAccess[i][12];
        Eigen::Vector3f mean(gmmAccess[i][1], gmmAccess[i][2], gmmAccess[i][3]);
        gaussians[i] = normal_random_variable(mean.cast<double>(), covar.cast<double>());
    }
    std::vector<Eigen::Vector3d> sampled(count);
#pragma omp parallel for
    for (int i = 0; i < count; ++i)
    {
        float r = (float)rand() / RAND_MAX;
        r /= accweights[N - 1];
        for (int j = 0; j < N; ++j)
        {
            if (accweights[j] >= r || j == N-1)
            {
                //std::cout << "Point " << i << " sampled from Gaussian " << j << ", r=" << r << std::endl;
                sampled[i] = gaussians[j]();
                break;
            }
        }
    }
    torch::Tensor pointcloud = torch::zeros({ count, 3 });
    for (int i = 0; i < count; ++i)
    {
        pointcloud.index_put_({ i, 0 }, (float)sampled[i].x());
        pointcloud.index_put_({ i, 1 }, (float)sampled[i].y());
        pointcloud.index_put_({ i, 2 }, (float)sampled[i].z());
    }
    return pointcloud;
}