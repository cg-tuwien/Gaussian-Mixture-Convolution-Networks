#include <algorithm>
#include <chrono>
#include <vector>
#include <stdio.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <torch/script.h>

#include <glm/glm.hpp>

#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "lbvh/query.h"
#include "lbvh/predicator.h"
#include "math/symeig_cuda.h"
