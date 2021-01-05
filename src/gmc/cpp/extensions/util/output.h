#ifndef GPE_UTIL_OUTPUT_H
#define GPE_UTIL_OUTPUT_H

#include "util/glm.h"
#include "util/gaussian.h"


template<int N_DIMS, typename scalar_t>
std::ostream& operator <<(std::ostream& stream, const gpe::Gaussian<N_DIMS, scalar_t>& g) {
    stream << "Gauss[" << g.weight << "; " << g.position[0];
    for (int i = 1; i < N_DIMS; i++)
        stream << "/" << g.position[i];
    stream << "; ";

    for (int i = 0; i < N_DIMS; i++) {
        for (int j = 0; j < N_DIMS; j++) {
            if (i != 0 || j != 0)
                stream << "/";
            stream << g.covariance[i][j];
        }
    }
    stream << "]";
    return stream;
}

template<typename scalar_t, int N_DIMS>
std::ostream& operator << (std::ostream& stream, const glm::vec<N_DIMS, scalar_t>& v) {
    stream << "vec" << N_DIMS << "(" << v[0];
    for (unsigned i = 1; i < N_DIMS; ++i) {
        stream << "/" << v[i];
    }
    stream << ")" << std::endl;
    return stream;
}

template<typename scalar_t, int N_DIMS>
std::ostream& operator << (std::ostream& stream, const glm::mat<N_DIMS, N_DIMS, scalar_t>& v) {
    stream << "mat" << N_DIMS << "(";
    for (unsigned i = 0; i < N_DIMS; ++i) {
        stream << v[i][0];
        for (unsigned j = 1; j < N_DIMS; ++j) {
            stream << ", " << v[i][j];
        }
        stream << " / ";
    }
    stream << ")" << std::endl;
    return stream;
}

namespace gpe {

template<typename scalar_t> EXECUTION_DEVICES
void printGaussian(const Gaussian<2, scalar_t>& g) {
    printf("g(w=%f, p=%f/%f, c=%f/%f//%f\n", g.weight, g.position.x, g.position.y, g.covariance[0][0], g.covariance[0][1], g.covariance[1][1]);
}
template<typename scalar_t> EXECUTION_DEVICES
void printGaussian(const Gaussian<3, scalar_t>& g) {
    printf("g(w=%f, p=%f/%f/%f, c=%f/%f/%f//%f/%f//%f\n",
           g.weight,
           g.position.x, g.position.y, g.position.z,
           g.covariance[0][0], g.covariance[0][1], g.covariance[0][2],
           g.covariance[1][1], g.covariance[1][2],
           g.covariance[2][2]);
}
}

#endif // GPE_UTIL_OUTPUT_H
