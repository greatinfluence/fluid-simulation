
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include <device_atomic_functions.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

#include "vendor/glm/glm.hpp"
#include "vendor/lodepng/lodepng.h"
#include "wob.cuh"
#include "random.cuh"

// B-S kernel
static __forceinline__ __host__ __device__ glm::vec3 G(glm::vec3 diff) {
#ifdef USE_2D
    return diff / (2 * PI) / glm::dot(diff, diff);
#else
    return diff / (4 * PI) / powf(glm::dot(diff, diff), 1.5f);
#endif // USE_2D
}

#ifndef USE_2D // for 3D only
// return the velocity at position x at time t;
static __forceinline__ __device__ glm::vec3 ComputeV(glm::vec3 x, glm::vec3* vorts, float* cdf, curandState& state) {
    // MC integration
    
    float totprob = cdf[lastind] - glm::length(vorts[TransformPos(x)]);
    if (totprob < 1e-5f) return glm::vec3(0);
	constexpr size_t nmc = 4096;
	glm::vec3 ans = glm::vec3(0);

    // take samples ~ ||wy||
    // G(rd) = d / 2pir in 2D, d / 4pir^2 in 3D
    // ans = sum cross(wy, G(x - y)) / p = sum cross(wy, G(x - y)) * sump / ||wy||
    for (size_t i = 0; i < nmc; ++i) {
        size_t ind = Sample(cdf, state);
        glm::vec3 y = glm::vec3(((ind % width) + 0.5f) / width, (((ind / width) % width) + 0.5f) / width,
                ((ind / width / width) + 0.5f) / width);
        while (glm::length(y - x) < 1e-5f) {
            ind = Sample(cdf, state);
            y = glm::vec3(((ind % width) + 0.5f) / width, (((ind / width) % width) + 0.5f) / width,
                ((ind / width / width) + 0.5f) / width);
        }
        glm::vec3 wy = vorts[ind];
        ans = ans + glm::cross(wy, G(x - y)) * totprob / glm::length(wy);
        //if (glm::length(glm::cross(wy, G(x - y))) * totprob / glm::length(wy) > 10000.f) {
        //    printf("%f %f %f - %f %f %f, %f %f\n", x.x, x.y, x.z, y.x, y.y, y.z, glm::length(wy), glm::length(G(x - y)));
       // }
    }
    ans /= nmc;
    //printf("%f %f %f: %f %f %f\n", x.x, x.y, x.z, ans.x, ans.y, ans.z);
    return ans;
}
#endif // USE_3D

// return the vorticity at position x at time t + 1;
static __forceinline__ __device__ void ComputeVortnDens(glm::vec3 x,
#ifndef USE_2D
    glm::vec3* velo, glm::vec3* nxvelo,
#endif // USE_3D
    glm::vec3* vorts, float* cdf, float* dens, curandState& state,
    glm::vec3& vort, float& den) {
    // semi-Lagrangian
    // w(t, x) = w(t - dt, x - dt int w(t - dt, y) X G(x - y)dy)
    
    //glm::vec3 v = ComputeV(x, vorts, cdf, state);
    
    // v = pPsi/px
    unsigned int ind = TransformPos(x);
#ifdef USE_2D // Advection (8)
    glm::vec3 v = glm::vec3(-vorts[ind].y, vorts[ind].x, 0);
#else // Advection (5)
    glm::vec3 v = ComputeV(x, vorts, cdf, state);
    nxvelo[ind] = v;
#endif // USE_2D

    glm::vec3 xa = x - dt * v;
    //float pPhipx = vort.x, pPhipy = vort.y;
    vort = glm::vec3(0);
    den = 0;
    constexpr unsigned int nb = 2048;
    float const sig = sqrtf(2 * nu * dt);
    for (unsigned int i = 0; i < nb; ++i) {
        //glm::vec3 xi = xa;
        glm::vec3 xi = Normaldist(xa, sig, state);
        //if(x.x > 0.4 && x.x < 0.6 && x.y > 0.4 && x.y < 0.6)
            //printf("%f %f %f: %f %f %f\n", xa.x, xa.y, xa.z, xi.x, xi.y, xi.z);
#ifdef USE_2D
		auto clmxi = glm::clamp(xi, glm::vec3(0.5f / width, 0.5f / width, 0),
						glm::vec3(1.f - 0.5f / width, 1.f - 0.5f / width, 0));
#else
		auto clmxi = glm::clamp(xi, glm::vec3(0.5f / width, 0.5f / width, 0.5f / width),
						glm::vec3(1.f - 0.5f / width, 1.f - 0.5f / width, 1.f - 0.5f / width));
#endif // USE_2D
        unsigned int ixi = TransformPos(clmxi);
#ifdef USE_2D
        if(xi.x > 0 && xi.y > 0 && xi.x < 1 && xi.y < 1)
            vort.z += vorts[ixi].z;
#else
        auto dx = .5f / width * vorts[ixi];
		auto clmxipdx = glm::clamp(xi + dx, glm::vec3(0.5f / width, 0.5f / width, 0.5f / width),
						glm::vec3(1.f - 0.5f / width, 1.f - 0.5f / width, 1.f - 0.5f / width));
		auto clmxisdx = glm::clamp(xi - dx, glm::vec3(0.5f / width, 0.5f / width, 0.5f / width),
						glm::vec3(1.f - 0.5f / width, 1.f - 0.5f / width, 1.f - 0.5f / width));
        glm::vec3 dv = velo[TransformPos(clmxipdx)] - velo[TransformPos(clmxisdx)];
        auto Dvw = glm::length(vorts[ixi]) * width * dv;
        //auto Dvw = glm::vec3(0);
        vort += vorts[ixi] + Dvw * dt;
#endif // USE_3D
        den += dens[ixi];
    }
#ifdef USE_2D
    vort.z /= nb;
#else
    vort /= nb;
#endif // USE_2D
    den /= nb;
    den = vorts[ind].x;
}

__global__ void initialize(curandState* states) {
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(19260817, ind, 0ull, &states[ind]);
}

#ifdef USE_2D
__global__ void computepsi(glm::vec3* vorts, glm::vec3* nxvorts) {
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%f %f\n", vorts[ind].x, nxvorts[ind].x);
    vorts[ind].x -= nxvorts[ind].x;
    vorts[ind].y -= nxvorts[ind].y;
}
#endif // USE_2D

__global__ void kernel(glm::vec3* vorts, float* cdf, float* dens,
                        glm::vec3* nxvorts,
#ifndef USE_2D
                        glm::vec3* velo, glm::vec3* nxvelo,
#endif // USE_3D
                        float* nxdens,
                        unsigned char* image,
                        curandState* states) {
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = ind % width;
#ifdef USE_2D
    unsigned int y = ind / width;
#else
    unsigned int y = (ind / width) % width;
    unsigned int z = ind / width / width;
#endif // USE_2D

    // Compute vorticity and density
    ComputeVortnDens(
#ifdef USE_2D
        glm::vec3((x + 0.5f) / width, (y + 0.5f) / width, 0),
#else
        glm::vec3((x + 0.5f) / width, (y + 0.5f) / width, (z + 0.5f) / width),
        velo, nxvelo,
#endif // USE_2D
        vorts,
        cdf,
        dens,
        states[ind],
        nxvorts[ind],
        nxdens[ind]
    );
}

// Output answer based on a shallow volumetric
__global__ void output(
    float* nxdens,
    unsigned char* image) {
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef USE_2D
    float ans =
        //    -vorts[ind].y;
            //nxvorts[ind].z;
            //vorts[ind].x * 20;
        nxdens[ind];
#else
    float ans = 0.f;
    // Return the average along the line
    for (unsigned int k = 0; k < width; ++k) {
        ans = ans + nxdens[ind + k * width * width];
    }
    ans /= width;

#endif // USE_2D
    if (ans > 0) {
        image[ind * 3] = glm::clamp(ans, 0.f, 1.f) * 255;
        image[ind * 3 + 2] = 0;
    }
    else {
        image[ind * 3] = 0;
        image[ind * 3 + 2] = glm::clamp(-ans, 0.f, 1.f) * 255;
    }
}

int main()
{
    const auto start = std::chrono::high_resolution_clock::now();

    unsigned char* image = new unsigned char[width * width * 3];
    memset(image, 0, sizeof(unsigned char) * width * width * 3);

    unsigned char* dimage = nullptr;
    checkCudaErrors(cudaMalloc(&dimage, sizeof(unsigned char) * width * width * 3));
    checkCudaErrors(cudaMemset(dimage, 0, sizeof(unsigned char) * width * width * 3));
    
    // Host Memory
    glm::vec3* vorts = nullptr;
    float* cdf = nullptr;
    float* dens = nullptr;
#ifdef USE_2D // Boundary condition only applicable in 2D
    Edge* edges = nullptr;
    float* ecdf = nullptr;
#endif // USE_2D

    // Device Memory
    glm::vec3* dvorts = nullptr;
    glm::vec3* dnxvorts = nullptr;
    float* dcdf = nullptr;
    float* ddens = nullptr;
    float* dnxdens = nullptr;
    curandState* dstates = nullptr;

#ifdef USE_2D // Boundary condition only applicable in 2D
    float* decdf = nullptr;
    Edge* dedges = nullptr;
#else
    glm::vec3* dvelo = nullptr;
    glm::vec3* dnxvelo = nullptr;
#endif // USE_2D

    constexpr size_t nedge = 4;
    {
        // Memory allocation
#ifdef USE_2D
        vorts = new glm::vec3[width * width];
        cdf = new float[width * width];
        dens = new float[width * width];
#else
        vorts = new glm::vec3[width * width * width];
        cdf = new float[width * width * width];
        dens = new float[width * width * width];
#endif // USE_2D
#ifdef USE_2D // Boundary condition only applicable in 2D
        edges = new Edge[nedge];
        ecdf = new float[nedge];
#endif // USE_2D
#ifdef USE_2D // Boundary condition only applicable in 2D
        memset(vorts, 0, sizeof(glm::vec3) * width * width);
        memset(dens, 0, sizeof(float) * width * width);

        checkCudaErrors(cudaMalloc(&dvorts, sizeof(glm::vec3) * width * width));
        checkCudaErrors(cudaMalloc(&dnxvorts, sizeof(glm::vec3) * width * width));
        checkCudaErrors(cudaMalloc(&dcdf, sizeof(float) * width * width));
        checkCudaErrors(cudaMalloc(&ddens, sizeof(float) * width * width));
        checkCudaErrors(cudaMalloc(&dnxdens, sizeof(float) * width * width));
        checkCudaErrors(cudaMalloc(&dstates, sizeof(curandState) * width * width));
#else
        memset(vorts, 0, sizeof(glm::vec3) * width * width * width);
        memset(dens, 0, sizeof(float) * width * width);

        checkCudaErrors(cudaMalloc(&dvorts, sizeof(glm::vec3) * width * width * width));
        checkCudaErrors(cudaMalloc(&dnxvorts, sizeof(glm::vec3) * width * width * width));
        checkCudaErrors(cudaMalloc(&dcdf, sizeof(float) * width * width * width));
        checkCudaErrors(cudaMalloc(&ddens, sizeof(float) * width * width * width));
        checkCudaErrors(cudaMalloc(&dnxdens, sizeof(float) * width * width * width));
        checkCudaErrors(cudaMalloc(&dstates, sizeof(curandState) * width * width * width));
        checkCudaErrors(cudaMalloc(&dvelo, sizeof(glm::vec3) * width * width * width));
        checkCudaErrors(cudaMalloc(&dnxvelo, sizeof(glm::vec3) * width * width * width));
        checkCudaErrors(cudaMemset(dnxvelo, 0, sizeof(glm::vec3) * width * width * width));
#endif // USE_2D
#ifdef USE_2D // Boundary condition only applicable in 2D
        checkCudaErrors(cudaMalloc(&dedges, sizeof(Edge) * nedge));
        checkCudaErrors(cudaMalloc(&decdf, sizeof(float) * nedge));
#endif // USE_2D
    }
    {
        // Data initialization
        
#ifdef USE_2D // Boundary condition only applicable in 2D
        edges[0] = {glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)};
        edges[1] = {glm::vec3(0, 1, 0), glm::vec3(1, 1, 0)};
        edges[2] = {glm::vec3(1, 1, 0), glm::vec3(1, 0, 0)};
        edges[3] = {glm::vec3(1, 0, 0), glm::vec3(0, 0, 0)};
        if(nedge == 8) {
            edges[4] = { glm::vec3(0.4f, 0.4f, 0), glm::vec3(0.6f, 0.4f, 0) };
            edges[5] = { glm::vec3(0.6f, 0.4f, 0), glm::vec3(0.6f, 0.6f, 0) };
            edges[6] = { glm::vec3(0.6f, 0.6f, 0), glm::vec3(0.4f, 0.6f, 0) };
            edges[7] = { glm::vec3(0.4f, 0.6f, 0), glm::vec3(0.4f, 0.4f, 0) };
        }
        for (unsigned int i = 0; i < nedge; ++i) {
            ecdf[i] = glm::length(edges[i].v2 - edges[i].v1) + (i != 0 ? ecdf[i - 1] : 0);
        }
        checkCudaErrors(cudaMemcpy(decdf, ecdf, sizeof(float) * nedge, cudaMemcpyHostToDevice));
#endif // USE_2D

        //glm::vec3 s1 = glm::vec3(256, 256, 0);

#ifdef USE_2D
        glm::vec3 s1 = glm::vec3(128, 128, 0);
        glm::vec3 s2 = glm::vec3(128, 384, 0);
        glm::vec3 s3 = glm::vec3(384, 128, 0);
        glm::vec3 s4 = glm::vec3(384, 384, 0);

        for (size_t i = 0; i < width; ++i)
            for (size_t j = 0; j < width; ++j) {
                glm::vec3 x = glm::vec3((i + 0.5f) / width, (j + 0.5f) / width, 0);
                if (x.x < 0.4f || x.y < 0.4f || x.x > 0.6f || x.y > 0.6f)

                    vorts[j * width + i] = glm::vec3(0, 0,
                        //0
                        //1e-3f / (1.f + fabs(i - s1.x))
                        //- 1e-3f / (1.f + fabs(i - s4.x))
                        - 1. / 250.f / (1.f + glm::length(glm::vec3(i, j, 0) - s1))
                        + 1. / 250.f / (1.f + glm::length(glm::vec3(i, j, 0) - s2))
                        // 1. / 250.f / (1.f + glm::length(glm::vec3(i, j, 0) - s3))
                        //+ 1e-3f / (1.f + glm::length(glm::vec3(i, j, 0) - s4))
                    );
                else vorts[j * width + i] = glm::vec3(0);
            }

        for (size_t i = 0; i < width; ++i)
            for (size_t j = 0; j < width;++j) {
                glm::vec3 x = glm::vec3((i + 0.5f) / width, (j + 0.5f) / width, 0);
                if (x.x < 0.4f || x.y < 0.4f || x.x > 0.6f || x.y > 0.6f)
                    //dens[j * width + i] = vorts[j * width + i].z * 1e4f;
                    //dens[j * width + i] = (vorts[j * width + i].z > 1e-5f) - (vorts[j * width + i].z < -1e-5f);
                    dens[j * width + i] = ((i / (width / 8) + j / (width / 8)) % 2) * 2.f - 1.f;
                else dens[j * width + i] = 0;
            }
#else
        glm::vec3 s1 = glm::vec3(48, 48, 48);
        glm::vec3 s2 = glm::vec3(128 - 48, 128 - 48, 128 - 48);
        glm::vec3 s3 = glm::vec3(384, 128, 128);
        glm::vec3 s4 = glm::vec3(384, 384, 128);

        for (size_t i = 0; i < width; ++i)
            for (size_t j = 0; j < width; ++j) 
				for(size_t k = 0; k < width; ++k) {
					glm::vec3 x = glm::vec3((i + 0.5f) / width, (j + 0.5f) / width, (k + 0.5f) / width);
						vorts[k * width * width + j * width + i] = glm::vec3(0,
							//0
							//1e-3f / (1.f + fabs(i - s1.x))
							//- 1e-3f / (1.f + fabs(i - s4.x))
							-1. / 2500.f / (1.f + glm::length(glm::vec3(i, j, k) - s1))
							+ 1. / 2500.f / (1.f + glm::length(glm::vec3(i, j, k) - s2))
							// 1. / 250.f / (1.f + glm::length(glm::vec3(i, j, 0) - s3))
							//+ 1e-3f / (1.f + glm::length(glm::vec3(i, j, 0) - s4))
						, 0);
                }

        for (size_t i = 0; i < width; ++i)
            for (size_t j = 0; j < width;++j)
                for(size_t k = 0; k < width; ++k) {
					glm::vec3 x = glm::vec3((i + 0.5f) / width, (j + 0.5f) / width, 0);
						//dens[j * width + i] = vorts[j * width + i].z * 1e4f;
						//dens[j * width + i] = (vorts[j * width + i].z > 1e-5f) - (vorts[j * width + i].z < -1e-5f);
                    dens[k * width * width + j * width + i] =
                            ((i / (width / 8)) % 2) * 2.f - 1.f;
            }

#endif // USE_2D
    }

    int block_size = 128;
    int picgrid_size = width * width / block_size;

    {
        // Preparation for the first frame
#ifdef USE_2D
        for (size_t ind = 0; ind <= lastind; ++ind) {
            if (dens[ind] > 0) image[ind * 3] = glm::clamp(dens[ind], 0.f, 1.f) * 255;
            else image[ind * 3 + 2] = glm::clamp(-dens[ind], 0.f, 1.f) * 255;
        }
#else
        cudaMemcpy(ddens, dens, sizeof(float)* width* width* width, cudaMemcpyHostToDevice);
        checkCudaErrors(cudaDeviceSynchronize());
        output<<<picgrid_size, block_size>>>(ddens, dimage);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(image, dimage, sizeof(unsigned char)* width* width * 3, cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize());
#endif // USE_2D

        {
            // Output the first frame
            char filename[100] = {};
            snprintf(filename, 100, "output\\Frame%04d.png", 0);
            auto errorcode = lodepng::encode(
                filename, image, width, width, LodePNGColorType::LCT_RGB);
            if (errorcode) {
                std::cerr << "error: " << lodepng_error_text(errorcode) << std::endl;
            }
        }
    }
#ifdef USE_2D
    int grid_size = width * width / block_size;
#else
    int grid_size = width * width * width / block_size;
#endif // USE_2D
#ifdef USE_2D // Boundary condition only applicable in 2D
    checkCudaErrors(cudaMemcpy(dedges, edges, sizeof(Edge)* nedge, cudaMemcpyHostToDevice));
#endif // USE_2D

    initialize <<<grid_size, block_size>>> (dstates);
    checkCudaErrors(cudaDeviceSynchronize());

    // Main loop
    for (int i = 0; i < nframe; ++i) {
        for (unsigned int ind = 0; ind <= lastind; ++ind) {
            cdf[ind] = glm::length(vorts[ind]) + (ind != 0 ? cdf[ind - 1] : 0);
        }
        printf("%f\n", cdf[lastind]);

#ifdef USE_2D
        checkCudaErrors(cudaMemcpy(
            dvorts, vorts, sizeof(glm::vec3) * width * width, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy(
            dcdf, cdf, sizeof(float) * width * width, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy(
            ddens, dens, sizeof(float) * width * width, cudaMemcpyHostToDevice
        ));
#else
        checkCudaErrors(cudaMemcpy(
            dvorts, vorts, sizeof(glm::vec3) * width * width * width, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy(
            dcdf, cdf, sizeof(float) * width * width * width, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy(
            ddens, dens, sizeof(float) * width * width * width, cudaMemcpyHostToDevice
        ));
        checkCudaErrors(cudaMemcpy(
            dvelo, dnxvelo, sizeof(glm::vec3) * width* width* width, cudaMemcpyDeviceToDevice
        ));
#endif // USE_2D
        checkCudaErrors(cudaDeviceSynchronize());

#ifdef USE_2D // Boundary condition only applicable in 2D
        // First, compute V0 and store it
        wob::v0computation <<<grid_size, block_size>>> (dvorts, dcdf, dnxvorts, dstates);
        checkCudaErrors(cudaDeviceSynchronize());

        std::cerr << "Done computing V0" << std::endl;

        // Then, compute Phi using WoB
        wob::wob <<<grid_size, block_size>>> (dvorts, dedges, nedge, decdf, dnxvorts, dstates, i * dt);
        checkCudaErrors(cudaDeviceSynchronize());

        std::cerr << "Done computing Phi" << std::endl;

        computepsi <<<grid_size, block_size>>> (dvorts, dnxvorts);
        checkCudaErrors(cudaDeviceSynchronize());

        std::cerr << "Done computing Psi" << std::endl;
#endif // USE_2D

        kernel <<<grid_size, block_size>>> (dvorts, dcdf, ddens, dnxvorts,
#ifndef USE_2D
            dvelo, dnxvelo,
#endif // USE_3D
            dnxdens, dimage, dstates);
        checkCudaErrors(cudaDeviceSynchronize());
        output <<<picgrid_size, block_size>>>(dnxdens, dimage);
        checkCudaErrors(cudaDeviceSynchronize());

        std::cerr << "Done computing everything" << std::endl;

#ifdef USE_2D
        checkCudaErrors(cudaMemcpy(
            vorts, dnxvorts, sizeof(glm::vec3) * width * width, cudaMemcpyDeviceToHost
        ));
        checkCudaErrors(cudaMemcpy(
            dens, dnxdens, sizeof(float) * width * width, cudaMemcpyDeviceToHost
        ));
#else
        checkCudaErrors(cudaMemcpy(
            vorts, dnxvorts, sizeof(glm::vec3) * width * width * width, cudaMemcpyDeviceToHost
        ));
        checkCudaErrors(cudaMemcpy(
            dens, dnxdens, sizeof(float) * width * width * width, cudaMemcpyDeviceToHost
        ));
#endif // USE_2D
        checkCudaErrors(cudaMemcpy(
            image, dimage, sizeof(unsigned char) * width * width * 3, cudaMemcpyDeviceToHost
        ));

        char filename[100] = {};
        snprintf(filename, 100, "output\\Frame%04d.png", i + 1);
        auto errorcode = lodepng::encode(
            filename, image, width, width, LodePNGColorType::LCT_RGB);
        if (errorcode) {
            std::cerr << "error: " << lodepng_error_text(errorcode) << std::endl;
        }

        if(i % fps == 0)
            std::cerr << "Done generating second " << (i / fps + 1) << std::endl;
        //std::cerr << vorts.cdf[lastind] << std::endl;
    }
    {
        // Memory clear
        checkCudaErrors(cudaFree(ddens));
        checkCudaErrors(cudaFree(dnxdens));
        checkCudaErrors(cudaFree(dvorts));
        checkCudaErrors(cudaFree(dnxvorts));
        checkCudaErrors(cudaFree(dcdf));
        checkCudaErrors(cudaFree(dimage));
#ifdef USE_2D
        checkCudaErrors(cudaFree(dedges));
        checkCudaErrors(cudaFree(decdf));
        delete[] ecdf;
        delete[] edges;
#else
        checkCudaErrors(cudaFree(dvelo));
        checkCudaErrors(cudaFree(dnxvelo));
#endif // USE_2D
        delete[] vorts;
        delete[] cdf;
        delete[] dens;
    }
    delete[] image;
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> diff = end - start;
    std::cerr << "Time usage: " << diff.count() << "s" << std::endl;

    return 0;
}
