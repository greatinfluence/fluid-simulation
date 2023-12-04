#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include "vendor/glm/glm.hpp"
#include "utils.cuh"

// Return the last position <= val
// Bin search, O(log(lastind))
__forceinline__ __host__ __device__ unsigned int lower_bound(float* arr, unsigned int r, float val) {
    unsigned int l = 0;
    while (l < r) {
        unsigned int mid = l + (r - l) / 2;
        if (arr[mid] > val) {
            r = mid;
        }
        else {
            l = mid + 1;
        }
    }
    return l;
}

// return a sample based on given cdf
__forceinline__ __device__ unsigned int Sample(float* cdf, curandState& state) {
    float val = curand_uniform(&state) * cdf[lastind];
    unsigned int ind = lower_bound(cdf, lastind + 1, val);
    return ind <= lastind ? ind : lastind;
}

// return a 2D random direction
__forceinline__ __device__ glm::vec3 Randdir2D(curandState& state) {
    float val = curand_uniform(&state) * 2 * PI;
    return glm::vec3(cosf(val), sinf(val), 0);
}

// return a 2D random direction that is in the same hemisphere with N
__forceinline__ __device__ glm::vec3 RanddirH2D(glm::vec3 N, curandState& state) {
    glm::vec3 D = Randdir2D(state);
    float cs = glm::dot(D, N);
    return cs < 0 ? D - 2 * cs * N : D;
}

// Return a sample ~ N(mu, sig^2 * I)
__forceinline__ __device__ glm::vec3 Normaldist(glm::vec3 mu, float sig, curandState& state) {
#ifdef USE_2D
    float2 sample = curand_normal2(&state);
    return glm::vec3(sample.x * sig + mu.x, sample.y * sig + mu.y, 0);
#else
    float2 sample = curand_normal2(&state);
    float s3 = curand_normal(&state);
    return glm::vec3(sample.x * sig + mu.x, sample.y * sig + mu.y, s3 * sig + mu.z);
#endif // USE_2D
}
