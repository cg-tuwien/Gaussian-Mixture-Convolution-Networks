
#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include "gmslib/PointSet.hpp"

float eval_rmse_psnr(torch::Tensor pointcloudSource, torch::Tensor pointcloudGenerated, bool psnr)
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
    std::vector<float> sqdiffs;
    sqdiffs.resize(nS);
    #pragma omp parallel for
    for (int i = 0; i < nS; ++i)
    {
        float minsqdiff = piG.nearestDistSearch(cpp_pcS[i]);
        sqdiffs[i] = minsqdiff;
        //if (i % 1000 == 0) std::cout << i << std::endl;
    }
    for (int i = 0; i < nS; ++i)
    {
        //std::cout << sqdiffs[i] << std::endl;
        summinsqdiffs += sqdiffs[i];
    }
    float avgminsqdiff = summinsqdiffs / nS;
    if (psnr) {
        float psnr = bboxPointsS.diagonal() / std::sqrt(avgminsqdiff);
        psnr = 20 * std::log10(psnr);
        return psnr;
    }
    else {
        return std::sqrt(avgminsqdiff) / bboxPointsS.diagonal();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_rmse_psnr", &eval_rmse_psnr);
}