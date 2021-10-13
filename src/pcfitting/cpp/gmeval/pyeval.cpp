
#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include <random>
#include "gmslib/pointset.hpp"

py::tuple eval_rmse_psnr(torch::Tensor pointcloudSource, torch::Tensor pointcloudGenerated, bool scaled, bool psnr)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
    #pragma omp parallel
    #pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloudSource = pointcloudSource.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    pointcloudGenerated = pointcloudGenerated.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloudSource.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(1) == 3, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloudSource.accessor<float, 2>();
    auto pGAccess = pointcloudGenerated.accessor<float, 2>();

    int nS = pointcloudSource.size(0);
    int nG = pointcloudGenerated.size(0);

    gms::PointSet cpp_pcS, cpp_pcG;
    cpp_pcS.reserve(size_t(nS));
    cpp_pcG.reserve(size_t(nG));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    for (unsigned i = 0; i < nG; ++i) {
        cpp_pcG.emplace_back(pGAccess[i][0], pGAccess[i][1], pGAccess[i][2]);
    }
    //gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());
    gms::PointIndex piG(cpp_pcG, std::numeric_limits<float>::max());

    //std::cout << "calculating bbox" << std::endl;

    gms::BBox bboxPointsS(cpp_pcS);

    //std::cout << "preprocessing done" << std::endl;

    float summinsqdiffs = 0;
    float summindiffs = 0;
    std::vector<float> sqdiffs;
    sqdiffs.resize(nS);
    #pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        float minsqdiff = piG.nearestDistSearch(cpp_pcS[i]);
        sqdiffs[i] = minsqdiff;
        //if (i % 1000 == 0) std::cout << i << std::endl;
    }
    float maxdiff = 0;
    //unsigned int now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    //std::cout << "LogID: " << std::to_string(now) << std::endl;
    //std::ofstream out("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/dists-" + std::to_string(now) + ".txt");
    for (int i = 0; i < nS; ++i)
    {
        //std::cout << sqdiffs[i] << std::endl;
        summinsqdiffs += sqdiffs[i];
        float diff = sqrt(sqdiffs[i]);
        summindiffs += diff;
        if (diff > maxdiff)
        {
            maxdiff = diff;
        }
        //out << diff << std::endl;
    }
   // out.close();
    float rmsd = std::sqrt(summinsqdiffs / nS);
    float averagediff = summindiffs / nS;
    float sumdeviations = 0;
    //float sumdeviationsM4 = 0;
    for (int i = 0; i < nS; ++i)
    {
        float deviation = sqrt(sqdiffs[i]) - averagediff;
        sumdeviations += pow(deviation, 2);
        //sumdeviationsM4 += pow(deviation, 4);
    }
    float standarddev = sqrt(sumdeviations / (nS - 1));
    //float moment4 = sumdeviationsM4 / nS;
    //float kurtosis = moment4 / pow(standarddev, 4);
    if (psnr) {
        float psnr = bboxPointsS.diagonal() / rmsd;
        psnr = 20 * std::log10(psnr);
        return py::make_tuple(psnr);
    }
    else {
        if (scaled)
        {
            return py::make_tuple(rmsd / bboxPointsS.diagonal(), averagediff / bboxPointsS.diagonal(), standarddev / bboxPointsS.diagonal(), maxdiff / bboxPointsS.diagonal());// , kurtosis);
        }
        return py::make_tuple(rmsd, averagediff, standarddev, maxdiff);//, kurtosis);
    }
}

py::tuple eval_rmsd_both_sides(torch::Tensor pointcloudSource, torch::Tensor pointcloudGenerated)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloudSource = pointcloudSource.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    pointcloudGenerated = pointcloudGenerated.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloudSource.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(1) == 3, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.size(1) == 3, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(0) == pointcloudGenerated.size(0), "point clouds must have same size")

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloudSource.accessor<float, 2>();
    auto pGAccess = pointcloudGenerated.accessor<float, 2>();

    int N = pointcloudSource.size(0);

    gms::PointSet cpp_pcS, cpp_pcG;
    cpp_pcS.reserve(size_t(N));
    cpp_pcG.reserve(size_t(N));
    for (unsigned i = 0; i < N; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    for (unsigned i = 0; i < N; ++i) {
        cpp_pcG.emplace_back(pGAccess[i][0], pGAccess[i][1], pGAccess[i][2]);
    }
    gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());
    gms::PointIndex piG(cpp_pcG, std::numeric_limits<float>::max());

    float summinsqdiffs_s = 0;
    float summinsqdiffs_g = 0;
    float summindiffs_s = 0;
    float summindiffs_g = 0;
    std::vector<float> diffs_s;
    std::vector<float> sqdiffs_s;
    std::vector<float> diffs_g;
    std::vector<float> sqdiffs_g;
    diffs_s.resize(N);
    sqdiffs_s.resize(N);
    diffs_g.resize(N);
    sqdiffs_g.resize(N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        float minsqdiff_s = piG.nearestDistSearch(cpp_pcS[i]);
        sqdiffs_s[i] = minsqdiff_s;
        diffs_s[i] = sqrt(minsqdiff_s);
        float minsqdiff_g = piS.nearestDistSearch(cpp_pcG[i]);
        sqdiffs_g[i] = minsqdiff_g;
        diffs_g[i] = sqrt(minsqdiff_g);
    }
    float maxdiff_s = 0;
    float maxdiff_g = 0;
    //unsigned int now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    //std::cout << "LogID: " << std::to_string(now) << std::endl;
    //std::ofstream out("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/dists-" + std::to_string(now) + ".txt");
    for (int i = 0; i < N; ++i)
    {
        //std::cout << sqdiffs[i] << std::endl;
        summinsqdiffs_s += sqdiffs_s[i];
        summindiffs_s += diffs_s[i];
        maxdiff_s = max(diffs_s[i], maxdiff_s);

        summinsqdiffs_g += sqdiffs_g[i];
        summindiffs_g += diffs_g[i];
        maxdiff_g = max(diffs_g[i], maxdiff_g);
        //out << diff << std::endl;
    }
    // out.close();
    float rmsd_s = std::sqrt(summinsqdiffs_s / N);
    float rmsd_g = std::sqrt(summinsqdiffs_g / N);
    float md_s = summindiffs_s / N;
    float md_g = summindiffs_g / N;
    float sumdeviations_s = 0;
    float sumdeviations_g = 0;
    for (int i = 0; i < N; ++i)
    {
        float deviation_s = diffs_s[i] - md_s;
        sumdeviations_s += pow(deviation_s, 2);
        float deviation_g = diffs_g[i] - md_g;
        sumdeviations_g += pow(deviation_g, 2);
    }
    float std_s = sqrt(sumdeviations_s / (N - 1));
    float std_g = sqrt(sumdeviations_g / (N - 1));
    return py::make_tuple(rmsd_s, md_s, std_s, maxdiff_s, rmsd_g, md_g, std_g, maxdiff_g);
}

py::tuple calc_rmsd_to_itself(torch::Tensor pointcloud)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloud = pointcloud.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloud.accessor<float, 2>();

    int nS = pointcloud.size(0);

    gms::PointSet cpp_pcS;
    cpp_pcS.reserve(size_t(nS));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());

    //std::cout << "calculating bbox" << std::endl;

    gms::BBox bboxPointsS(cpp_pcS);

    //std::cout << "preprocessing done" << std::endl;

    float summinsqdiffs = 0;
    float summindiffs = 0;
    std::vector<float> sqdiffs;
    sqdiffs.resize(nS);
#pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        float minsqdiff = piS.nearestDistSearch(cpp_pcS[i], i);
        sqdiffs[i] = minsqdiff;
        //if (i % 1000 == 0) std::cout << i << std::endl;
    }
    for (int i = 0; i < nS; ++i)
    {
        //std::cout << sqdiffs[i] << std::endl;
        summinsqdiffs += sqdiffs[i];
        float diff = sqrt(sqdiffs[i]);
        summindiffs += diff;
    }
    float avgminsqdiff = summinsqdiffs / nS;
    return py::make_tuple(std::sqrt(avgminsqdiff), summindiffs / nS);
}

py::tuple cov_measure(torch::Tensor pointcloud)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloud = pointcloud.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloud.accessor<float, 2>();

    int nS = pointcloud.size(0);

    gms::PointSet cpp_pcS;
    cpp_pcS.reserve(size_t(nS));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());

    //std::cout << "calculating bbox" << std::endl;

    gms::BBox bboxPointsS(cpp_pcS);

    //std::cout << "preprocessing done" << std::endl;

    float summindiffs = 0;
    std::vector<float> diffs;
    diffs.resize(nS);
#pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        float minsqdiff = piS.nearestDistSearch(cpp_pcS[i], i);
        diffs[i] = sqrt(minsqdiff);
    }
    for (int i = 0; i < nS; ++i)
    {
        summindiffs += diffs[i];
    }
    float avgmindiff = summindiffs / nS;
    float std = 0;
    for (int i = 0; i < nS; ++i)
    {
        std += pow(diffs[i] - avgmindiff, 2);
    }
    std /= nS;
    std = sqrt(std);
    return py::make_tuple(std / avgmindiff, std);
}

py::tuple cov_measure_5(torch::Tensor pointcloud)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloud = pointcloud.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloud.accessor<float, 2>();

    int nS = pointcloud.size(0);

    gms::PointSet cpp_pcS;
    cpp_pcS.reserve(size_t(nS));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());

    //std::cout << "calculating bbox" << std::endl;

    gms::BBox bboxPointsS(cpp_pcS);

    //std::cout << "preprocessing done" << std::endl;

    float summindiffs = 0;
    std::vector<float> diffs;
    diffs.resize(nS);
#pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        std::vector<gms::uint> knn;
        piS.annSearch(cpp_pcS[i], 5, knn);
        float sum = 0;
        for (int j = 0; j < 5; ++j)
        {
            sum += gms::dist(cpp_pcS[i], cpp_pcS[j]);
        }
        sum /= 5;
        diffs[i] = sum;
    }
    for (int i = 0; i < nS; ++i)
    {
        summindiffs += diffs[i];
    }
    float avgmindiff = summindiffs / nS;
    float std = 0;
    for (int i = 0; i < nS; ++i)
    {
        std += pow(diffs[i] - avgmindiff, 2);
    }
    std /= nS;
    std = sqrt(std);
    return py::make_tuple(std / avgmindiff, std);
}

py::tuple calc_std_1_5(torch::Tensor pointcloudSource, torch::Tensor pointcloudGenerated)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloudSource = pointcloudSource.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    pointcloudGenerated = pointcloudGenerated.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloudSource.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudSource.size(1) == 3, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloudGenerated.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloudSource.accessor<float, 2>();
    auto pGAccess = pointcloudGenerated.accessor<float, 2>();

    int nS = pointcloudSource.size(0);
    int nG = pointcloudGenerated.size(0);

    gms::PointSet cpp_pcS, cpp_pcG;
    cpp_pcS.reserve(size_t(nS));
    cpp_pcG.reserve(size_t(nG));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    for (unsigned i = 0; i < nG; ++i) {
        cpp_pcG.emplace_back(pGAccess[i][0], pGAccess[i][1], pGAccess[i][2]);
    }
    //gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());
    gms::PointIndex piG(cpp_pcG, std::numeric_limits<float>::max());

    //std::cout << "calculating bbox" << std::endl;

    gms::BBox bboxPointsS(cpp_pcS);

    //std::cout << "preprocessing done" << std::endl;

    std::vector<float> diffs1;
    diffs1.resize(nS);
    std::vector<float> diffs5;
    diffs5.resize(nS);
    gms::uint K = 20;
#pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        float minsqdiff = piG.nearestDistSearch(cpp_pcS[i]);
        diffs1[i] = sqrt(minsqdiff);
        std::vector<float> knn = piG.nearestKDistSearch(cpp_pcS[i], K);
        //piG.annSearch(cpp_pcS[i], 5, knn);
        float sum = 0;
        for (int j = 0; j < K; ++j)
        {
            sum += knn[j];
        }
        sum /= K;
        diffs5[i] = sum;
    }
    unsigned int now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    //std::cout << "LogID: " << std::to_string(now) << std::endl;
    //std::ofstream out1("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/dists-1-" + std::to_string(now) + ".txt");
    //std::ofstream out5("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/dists-5-" + std::to_string(now) + ".txt");
    float sumdiffs1 = 0;
    float sumdiffs5 = 0;
    for (int i = 0; i < nS; ++i)
    {
        //std::cout << sqdiffs[i] << std::endl;
        sumdiffs1 += diffs1[i];
        sumdiffs5 += diffs5[i];
        //out1 << diffs1[i] << std::endl;
        //out5 << diffs5[i] << std::endl;
    }
    // out.close();
    float m1 = sumdiffs1 / nS;
    float m5 = sumdiffs5 / nS;
    float sumdeviations1 = 0;
    float sumdeviations5 = 0;
    for (int i = 0; i < nS; ++i)
    {
        float deviation1 = diffs1[i] - m1;
        sumdeviations1 += pow(deviation1, 2);
        float deviation5 = diffs5[i] - m5;
        sumdeviations5 += pow(deviation5, 2);
    }
    float std1 = sqrt(sumdeviations1 / (nS - 1));
    float std5 = sqrt(sumdeviations5 / (nS - 1));
    return py::make_tuple(std1, std1 / m1, std5, std5 / m5);
}

torch::Tensor nn_graph(torch::Tensor pointcloud, int ncount)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloud = pointcloud.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloud.accessor<float, 2>();

    int nS = pointcloud.size(0);

    gms::PointSet cpp_pcS;
    cpp_pcS.reserve(size_t(nS));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());

    std::vector<std::vector<size_t>> W(nS, std::vector<size_t>(ncount, 0));

#pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        std::vector<size_t> nearest;
        std::vector<float> knn = piS.nearestKDistSearch(cpp_pcS[i], ncount, i, &nearest);
        for (int j = 0; j < ncount; ++j)
        {
            //W[i][nearest[j]] = 1;
            //W[nearest[j]][i] = 1;
            W[i][j] = nearest[j];
        }
    }

    torch::Tensor result = torch::zeros({ nS, ncount }, torch::TensorOptions().dtype(torch::kInt64));
    for (int i = 0; i < nS; ++i)
    {
        for (int j = 0; j < ncount; ++j)
        {
            result.index_put_({ i, j }, (int64_t)W[i][j]);
        }
    }
    return result;
}

torch::Tensor nn_graph_sub(torch::Tensor pointcloud, int samplecount, int ncount)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    //std::cout << "Start" << std::endl;
    pointcloud = pointcloud.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.sizes().size() == 2, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(0) > 0, "point cloud must have dimensions of N x 3");
    TORCH_CHECK(pointcloud.size(1) == 3, "point cloud must have dimensions of N x 3");

    //std::cout << "Start Preprocessing" << std::endl;

    auto pSAccess = pointcloud.accessor<float, 2>();

    int nS = pointcloud.size(0);

    gms::PointSet cpp_pcS;
    cpp_pcS.reserve(size_t(nS));
    for (unsigned i = 0; i < nS; ++i) {
        cpp_pcS.emplace_back(pSAccess[i][0], pSAccess[i][1], pSAccess[i][2]);
    }
    gms::PointIndex piS(cpp_pcS, std::numeric_limits<float>::max());

    std::vector<size_t> indizes(nS);
    for (int i = 0; i < nS; ++i)
    {
        indizes[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indizes.begin(), indizes.end(), g);

    std::vector<std::vector<size_t>> W(samplecount, std::vector<size_t>(1 + ncount, 0));

#pragma omp parallel for
    for (int i = 0; i < samplecount; ++i)
    {
        W[i][0] = indizes[i];
        std::vector<size_t> nearest;
        std::vector<float> knn = piS.nearestKDistSearch(cpp_pcS[indizes[i]], ncount, indizes[i], &nearest);
        for (int j = 0; j < ncount; ++j)
        {
            //W[i][nearest[j]] = 1;
            //W[nearest[j]][i] = 1;
            W[i][j + 1] = nearest[j];
        }
    }

    torch::Tensor result = torch::zeros({ samplecount, ncount + 1 }, torch::TensorOptions().dtype(torch::kInt64));
    for (int i = 0; i < samplecount; ++i)
    {
        for (int j = 0; j <= ncount; ++j)
        {
            result.index_put_({ i, j }, (int64_t)W[i][j]);
        }
    }
    return result;
}

//metric from locally consistent gmm - checks smoothness of responsibilities - not what we need!
float smoothnes(torch::Tensor responsibilities, torch::Tensor nngraph)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    ;

    auto respAccess = responsibilities.accessor<double, 2>();
    auto nnGAccess = nngraph.accessor<int64_t, 2>();

    int nP = responsibilities.size(0);
    int nG = responsibilities.size(1);
    int nN = nnGAccess.size(1);

    std::cout << "Starting calculation" << std::endl;

    float R = 0;

    #pragma omp parallel for
    for (int i = 0; i < nP; ++i)
    {
        for (int j = 0; j < nP; ++j)
        {
            bool w = false;
            for (int n = 0; n < nN; ++n)
            {
                w |= (nnGAccess[i][n] == j);
                if (w) break;
                w |= (nnGAccess[j][n] == i);
                if (w) break;
            }
            if (w)
            {
                double kldiv = 0;
                for (int k = 0; k < nG; ++k)
                {
                    if (respAccess[i][k] > 0 && respAccess[j][k] > 0)
                    {
                        kldiv += respAccess[i][k] * (log(respAccess[i][k]) - log(respAccess[j][k]));
                        kldiv += respAccess[j][k] * (log(respAccess[j][k]) - log(respAccess[i][k]));
                    }
                }
                R += kldiv;
            }
        }
        //if (i % 10 == 0) std::cout << i << std::endl;
    }
    R *= 0.5;
    return R;
}

//densities (N)
//nngraph (N, k)
float irregularity(torch::Tensor densities, torch::Tensor nngraph)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    ;

    auto dsAccess = densities.accessor<float, 1>();
    auto nnGAccess = nngraph.accessor<int64_t, 2>();

    int nP = dsAccess.size(0);
    int nN = nnGAccess.size(1);

    std::cout << "Starting calculation" << std::endl;

    std::vector<float> stdevs;
    stdevs.resize(nP);

    #pragma omp parallel for
    for (int i = 0; i < nP; ++i)
    {
        float dI = dsAccess[i];
        float mean = dI;
        for (int k = 0; k < nN; ++k)
        {
            int64_t j = nnGAccess[i][k];
            float dJ = dsAccess[j];
            mean += dJ;
        }
        mean /= (nN + 1);

        float sumdev2 = 0;
        /*if (i == 65525)
        {
            std::cout << dI << std::endl;
            std::cout << "--" << std::endl;
        }*/
        //std::vector<float> locals(nN);
        sumdev2 += std::pow(dI - mean, 2);
        for (int k = 0; k < nN; ++k)
        {
            int64_t j = nnGAccess[i][k];
            float dJ = dsAccess[j];
            //sumdev2 += std::pow((dJ - dI) / dI, 2);
            //sumdev2 += std::abs(dJ - dI) / dI;
            //sumdev2 += std::abs(dJ - dI);
            //locals[k] = std::abs(dJ - dI) / dI;
            //if (i == 65525) std::cout << dJ << std::endl;
            sumdev2 += std::pow(dJ - mean, 2);

        }
        //stdevs[i] = std::sqrt(sumdev2 / nN);
        //stdevs[i] = sumdev2 / nN;
        //stdevs[i] = (sumdev2 / (dI * nN));
        /*if (i == 65525) {
            std::cout << "--" << std::endl;
            std::cout << stdevs[i] << std::endl;
            std::cout << "---" << std::endl;
        }*/
        //std::sort(locals.begin(), locals.end());
        //stdevs[i] = locals[nN / 2];
        stdevs[i] = std::sqrt(sumdev2 / (nN)) / mean;   //nN, not +1 because std bias
    }
    /*unsigned int now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "LogID: " << std::to_string(now) << std::endl;
    std::ofstream out("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/sm-" + std::to_string(now) + ".txt");*/
    float sumstdevs = 0;
    for (int i = 0; i < nP; ++i)
    {
        sumstdevs += stdevs[i];
        //out << stdevs[i] << std::endl;
    }
    //out.close();
    float result = sumstdevs / nP;

    //std::sort(stdevs.begin(), stdevs.end());
    //float result = stdevs[nP / 2];


    return result;
}

//densities (N)
//nngraph (S, k)
py::tuple irregularity_sub(torch::Tensor densities, torch::Tensor nngraph_sub)
{
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel
#pragma omp master

    ;

    auto dsAccess = densities.accessor<float, 1>();
    auto nnGAccess = nngraph_sub.accessor<int64_t, 2>();

    int nP = dsAccess.size(0);
    int nS = nnGAccess.size(0);
    int nN = nnGAccess.size(1);

    std::cout << "Starting calculation" << std::endl;

    std::vector<float> stdevs;
    stdevs.resize(nS);

    #pragma omp parallel for
    for (int sampleIndex = 0; sampleIndex < nS; ++sampleIndex)
    {
        float mean = 0;
        for (int k = 0; k < nN; ++k)
        {
            int64_t j = nnGAccess[sampleIndex][k];
            float dJ = dsAccess[j];
            mean += dJ;
        }
        mean /= nN;

        //int64_t i = nnGAccess[sampleIndex][0];
        //float dI = dsAccess[i];

        float sumdev2 = 0;
        for (int k = 0; k < nN; ++k)
        {
            int64_t j = nnGAccess[sampleIndex][k];
            float dJ = dsAccess[j];
            sumdev2 += std::pow(dJ - mean, 2);
            //sumdev2 += std::abs(dJ - dI) / dI;

        }
        stdevs[sampleIndex] = std::sqrt(sumdev2 / (nN - 1)) / mean; 
        //stdevs[sampleIndex] = sumdev2 / (nN - 1);
    }
    /*unsigned int now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "LogID: " << std::to_string(now) << std::endl;
    std::ofstream out("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/EvalLogs/sm-" + std::to_string(now) + ".txt");*/
    float sumcvs = 0;
    for (int i = 0; i < nS; ++i)
    {
        sumcvs += stdevs[i];
        //out << stdevs[i] << std::endl;
    }
    //out.close();
    //float result = sumstdevs / nS;
    float mean = sumcvs / nS;
    float sumdevs = 0;
    for (int i = 0; i < nS; ++i)
    {
        sumdevs += std::pow(stdevs[i] - mean, 2);
    }
    float cvstd = std::sqrt(sumdevs / (nS - 1));
    return py::make_tuple(mean, cvstd);

    //std::sort(stdevs.begin(), stdevs.end());
    //float result = stdevs[nS / 2];


    //return result;
}


#include "sampler.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_rmse_psnr", &eval_rmse_psnr);
    m.def("calc_rmsd_to_itself", &calc_rmsd_to_itself);
    m.def("cov_measure", &cov_measure);
    m.def("sample_gmm", &sample_gmm);
    m.def("eval_rmsd_both_sides", &eval_rmsd_both_sides);
    m.def("calc_std_1_5", &calc_std_1_5);
    m.def("cov_measure_5", &cov_measure_5);
    m.def("avg_kl_div", &avg_kl_div);
    m.def("nn_graph", &nn_graph);
    m.def("nn_graph_sub", &nn_graph_sub);
    m.def("smoothness", &smoothnes);
    m.def("irregularity", &irregularity);
    m.def("irregularity_sub", &irregularity_sub);
}