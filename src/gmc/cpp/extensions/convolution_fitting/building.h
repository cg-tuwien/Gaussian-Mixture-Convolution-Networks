//The MIT License (MIT)

//Copyright (c) 2019 Toru Niina

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.

//(based on https://github.com/ToruNiina/lbvh)

#ifndef LBVH_BUILDING_H
#define LBVH_BUILDING_H

#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "util/scalar.h"

namespace lbvh {

namespace kernels {
template<typename UInt>
__host__ __device__
inline uint2 determine_range(UInt const* node_code, const unsigned int num_leaves, unsigned int idx)
{
#ifndef GPE_MORTON_OLD_WAY
    auto nodeCode = [&node_code](unsigned int idx){ return (node_code[idx] & 0xFFFFFFFF00000000ULL) + idx; };
#else
    auto nodeCode = [&node_code](unsigned int idx){ return node_code[idx]; };
#endif

    if(idx == 0)
    {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    const UInt self_code = nodeCode(idx);
    const int L_delta = gpe::common_upper_bits(self_code, nodeCode(idx-1));
    const int R_delta = gpe::common_upper_bits(self_code, nodeCode(idx+1));
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range
    const int delta_min = gpe::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = gpe::common_upper_bits(self_code, nodeCode(i_tmp));
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = gpe::common_upper_bits(self_code, nodeCode(i_tmp));
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = gpe::common_upper_bits(self_code, nodeCode(i_tmp));
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0)
    {
        gpe::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

template<typename UInt>
__host__ __device__
inline unsigned int find_split(UInt const* node_code, const unsigned int num_leaves,
               const unsigned int first, const unsigned int last) noexcept
{
#ifndef GPE_MORTON_OLD_WAY
    auto nodeCode = [&node_code](unsigned int idx){ return (node_code[idx] & 0xFFFFFFFF00000000ULL) + idx; };
#else
    auto nodeCode = [&node_code](unsigned int idx){ return node_code[idx]; };
#endif

    const UInt first_code = nodeCode(first);
    const UInt last_code  = nodeCode(last);
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = gpe::common_upper_bits(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = gpe::common_upper_bits(first_code, nodeCode(middle));
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}

} // namespace kernels

} // namespace lbvh


#endif // LBVH_BUILDING_H
