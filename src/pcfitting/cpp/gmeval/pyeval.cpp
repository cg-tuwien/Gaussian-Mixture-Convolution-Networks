
#include <torch/extension.h>
#include <vector>
#include <omp.h>
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

#include "sampler.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_rmse_psnr", &eval_rmse_psnr);
    m.def("calc_rmsd_to_itself", &calc_rmsd_to_itself);
    m.def("cov_measure", &cov_measure);
    m.def("sample_gmm", &sample_gmm);
    m.def("eval_rmsd_both_sides", &eval_rmsd_both_sides);
}