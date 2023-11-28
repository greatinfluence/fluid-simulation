#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vendor/glm/glm.hpp"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define USE_2D

constexpr int fps = 120; // fps
constexpr float dt = 1.f / fps; // 1 / fps

constexpr float PI = 3.14159265358979323f; // pi
constexpr float nu = 1e-5f; // Viscosity
constexpr int nframe = fps * 1; // Number of frames

#ifdef USE_2D // 2D setting
constexpr unsigned int width = 512; // width
constexpr unsigned int lastind = width * width - 1; // the last valid index
#else // 3D setting
constexpr unsigned int width = 128; // width
constexpr unsigned int lastind = width * width * width - 1; // The last valid index
#endif // USE_2D

constexpr unsigned int buffersize = 8192; // Buffer size used by wob

// Transform position to index
static __forceinline__ __host__ __device__ unsigned int TransformPos(glm::vec3 x) {
#ifdef USE_2D
    unsigned int ind = fminf(roundf(x.y * width - 0.5f), width - 1) * width +
        fminf(roundf(x.x * width - 0.5f), width - 1);
#else
    unsigned int ind = fminf(roundf(x.z * width - 0.5f), width - 1) * width * width +
        fminf(roundf(x.y * width - 0.5f), width - 1) * width +
        fminf(roundf(x.x * width - 0.5f), width - 1);
#endif // USE_2D
    return ind;
}

static __forceinline__ __host__ __device__ unsigned int FixPosition(glm::vec3 x, glm::vec3 normal) {
    int xpos = (x.x - 0.5f) * width, ypos = (x.y - 0.5f) * width;
    if (xpos < 0) xpos = 0;
    else if (xpos > width - 1) xpos = width - 1;
    else {
        if (normal.x > 0) xpos = xpos + 1;
    }
    if (ypos < 0) ypos = 0;
    else if (ypos > width - 1) ypos = width - 1;
    else {
        if (normal.y > 0) ypos = ypos + 1;
    }
    return ypos * width + xpos;
}
