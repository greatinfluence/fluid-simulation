#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "vendor/glm/glm.hpp"

#include "random.cuh"

#ifdef USE_2D // This module is for 2D rendering only
struct Edge {
    glm::vec3 v1, v2;
};

namespace wob {
    // The assigned boundary value of the stream function
    static __forceinline__ __device__ float b(glm::vec3 pos, float t) {
        return 0;
        if (pos.x < 0.1f) {
            if (pos.y > 0.6f) return 0.01f;
            else if (pos.y < 0.4f) return -0.01f;
            else return -0.01f + (pos.y - 0.4f) * 0.02f / 0.2f;
        }
        else {
            if (pos.y > 0.2f) return 0.01f;
            else if (pos.y < 0.1f) return -0.01f;
            else return -0.01f + (pos.y - 0.1f) * 0.02f / 0.1f;

        }
    }

	struct Ray {
		glm::vec3 pos, dir;
	};

	struct HitInfo {
		glm::vec3 hitpos; // Hit position
		glm::vec3 normal; // Normal at the hit
		float hitdist; // Distance between the hit point and the starting point
	};

    constexpr float mindist = 1e-4f;

    static __forceinline__ __host__ __device__ void TestHit(Edge* edges, size_t const nedge, Ray ray, HitInfo& hitinfo) {
        for (size_t i = 0; i < nedge; ++i) {
            glm::vec3 const ed = edges[i].v2 - edges[i].v1;
            glm::vec3 const edhat = glm::normalize(ed);
            glm::vec3 const dperp = ray.dir - glm::dot(edhat, ray.dir) * edhat;
            if (glm::length(dperp) < 1e-6f) continue; // Parallel
            float const t = glm::dot(dperp, edges[i].v1 - ray.pos) / glm::dot(dperp, dperp);
            if (t > mindist && t < hitinfo.hitdist) {
                glm::vec3 const hitpos = ray.pos + t * ray.dir;
                if (glm::dot(edges[i].v1 - hitpos, edges[i].v2 - hitpos) <= 0) {
					hitinfo.hitdist = t;
                    hitinfo.hitpos = glm::dot(hitpos - edges[i].v1, edhat) * edhat + edges[i].v1;
					hitinfo.normal = glm::vec3(edhat.y, -edhat.x, 0);
                }
            }
        }
    }

    // Return the core G
    static __forceinline__ __host__ __device__ float Gp(glm::vec3 x) {
        return log(glm::length(x)) / 2 / PI;
    }

    // Return the normal derivative of G
    static __forceinline__ __host__ __device__ float pGpny(glm::vec3 x, glm::vec3 ny) {
        return (x.x * ny.x + x.y * ny.y) / (x.x * x.x + x.y * x.y) / 2 / PI;
    }

    // Return the normal derivative of G
    static __forceinline__ __host__ __device__ float pGpxk(glm::vec3 x, unsigned int dim) {
        return x[dim] / (x.x * x.x + x.y * x.y) / 2 / PI;
    }
    
    // Return the partial derivative of G relative to xk and ny
    static __forceinline__ __host__ __device__ float p2Gpxkpny(glm::vec3 x, glm::vec3 ny, unsigned int dim) {
#ifdef USE_2D
        float x2 = x.x * x.x, y2 = x.y * x.y;
        if (dim == 0) {
            // dx
            return (-ny.x * x2 - 2 * ny.y * x.x * x.y + ny.x * y2) / (x2 + y2) / (x2 + y2) / 2 / PI;
        }
        else {
            // dy
            return (-ny.y * y2 - 2 * ny.x * x.y * x.x + ny.y * x2) / (x2 + y2) / (x2 + y2) / 2 / PI;
        }
#else
#endif // USE_2D
    }

    // return V0 at position x at time t
    static __forceinline__ __device__ void ComputeV0(glm::vec3 x, glm::vec3* vorts, float* cdf, glm::vec3& output, curandState& state) {
        float ansx = 0, ansy = 0, ansz = 0;
        constexpr unsigned int nmc = 16384;
        unsigned int ix = TransformPos(x);
        glm::vec3 wx = vorts[ix];
        if (cdf[lastind] - glm::length(wx) < 1e-5f) {
            output.x = 0;
            output.y = 0;
            output.z = 0;
        }
        for (unsigned int i = 0; i < nmc; ++i) {
            size_t ind = Sample(cdf, state);
            while (ind == ix) {
                ind = Sample(cdf, state);
            }
            glm::vec3 y = glm::vec3(((ind % width) + 0.5f) / width, ((ind / width) + 0.5f) / width, 0);
            // glm::vec3 wy = vorts[ind];
            ansx += pGpxk(x - y, 0) * (cdf[lastind] - wx.z) * (vorts[ind].z > 0 ? 1.f : -1.f); // Gp(x - y) * wy / (wy / (cdf - wx)) = Gp(x - y) * (cdf - wx)
            ansy += pGpxk(x - y, 1) * (cdf[lastind] - wx.z) * (vorts[ind].z > 0 ? 1.f : -1.f); // Gp(x - y) * wy / (wy / (cdf - wx)) = Gp(x - y) * (cdf - wx)
            ansz += Gp(x - y) * (cdf[lastind] - wx.z) * (vorts[ind].z > 0 ? 1.f : -1.f);
            //printf("%f %f %f -> %f %f %f: %f\n", x.x, x.y, x.z, y.x, y.y, y.z, Gp(x - y));
        }
        output.x = ansx / nmc;
        output.y = ansy / nmc;
        output.z = ansz / nmc;
    }

    __forceinline__ __global__ void v0computation(
        glm::vec3* vorts, float* cdf,
        glm::vec3* nxvorts, curandState* states) {
        unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = ind / width;
        unsigned int x = ind % width;

        // Compute V0 and store in nxvorts[ind].z
        ComputeV0(glm::vec3((x + 0.5f) / width, (y + 0.5f) / width, 0), vorts, cdf, nxvorts[ind], states[ind]);
    }

    struct Status {
        unsigned int depth;
        float coef;
        glm::vec3 dir;
        glm::vec3 pos;
        glm::vec3 normal;
    };

    struct StatStack {
        __device__ StatStack(Status* _buffer) :
            data{ _buffer }, cap{ buffersize / sizeof(Status) }, sz{ 0u } {}

        __forceinline__ __device__ Status& back() const {
            return data[sz - 1];
        }

        __forceinline__ __device__ Status& Getcontent(unsigned int pos) const {
            return data[pos];
        }
        
        __forceinline__ __device__ bool empty() const {
            return sz == 0;
        }

        __forceinline__ __device__ void pop_back() {
            --sz;
        }
    
        __forceinline__ __device__ void push_back(Status&& v) {
            data[sz++] = v;
        }

        Status* data; // The data of the stack
        unsigned int cap; // The capacity of the stack
        unsigned int sz; // The used size of the stack
    };

    static __forceinline__ __device__ bool insidetest(glm::vec3 pos, glm::vec3 const dir,
        Edge* edges, unsigned int nedge) {
        bool hitcount = false;
        //printf("Inside test begin\n");
        int debug = 0;
        if (fabs(pos.x - 0.915280) + fabs(pos.y - 1.000000) < 1e-4f &&
            fabs(dir.x + 0.715122) + fabs(dir.x + 0.698999) < 1e-4f) {
            debug = 1;
        }
        if (debug) printf("%f %f %f -> %f %f %f\n", pos.x, pos.y, pos.z, dir.x, dir.y, dir.z);
        while (true) {
            //printf("Inside test: %f %f %f\n", pos.x, pos.y, pos.z);
			HitInfo hi;
			hi.hitdist = 1e10f;
			TestHit(edges, nedge, Ray{ pos, dir }, hi);
            if (hi.hitdist < 9e9f) {
                hitcount = !hitcount;
                if(debug) printf("%f %f %f: %d\n", hi.hitpos.x, hi.hitpos.y, hi.hitpos.z, debug ++);
                pos = hi.hitpos;
            }
            else break;
        }
        //printf("Inside test end\n");
        return hitcount;
    }

    static __forceinline__ __device__ bool isInside(glm::vec3 pos,
        Edge* edges, unsigned int nedge, curandState& state) {
        glm::vec3 dir = Randdir2D(state);
        bool outcome = insidetest(pos, dir, edges, nedge);
                    //if (fabs(pos.x - 0.938477) + fabs(pos.y - 0.030273) < 1e-3f) {
                     //   printf("%f %f %f -> %f %f %f, %d\n", pos.x, pos.y, pos.z, dir.x, dir.y, dir.z, outcome);
                   // }
                    return outcome;
    }

    static __forceinline__ __device__ void GetVal(glm::vec3 pos, float t, glm::vec3* nxvorts, glm::vec3 normal, float& output) {
        unsigned int clmpedind = FixPosition(pos, normal);
        //glm::vec3 dir = glm::vec3(-normal.y, normal.x, 0);
        float bdval = b(pos, t);
        //float bdnval = b(pos + dir * 1e-4f, t);
        //float dvdx = (bdnval - bdval) * 1e4f;
        //output.x = nxvorts[clmpedind].x + dvdx * dir.x;
        //output.y = nxvorts[clmpedind].y + dvdx * dir.y;
        output = nxvorts[clmpedind].z + bdval;
    }

    static __forceinline__ __device__ void SampleEdge(Edge* edges, unsigned int nedge, float* ecdf, HitInfo& output, curandState& state) {
        float val = curand_uniform(&state) * ecdf[nedge - 1];
        unsigned int ind = lower_bound(ecdf, nedge, val);
        if (ind >= nedge) ind = nedge - 1;
        glm::vec3 dir = glm::normalize(edges[ind].v2 - edges[ind].v1);
        output.hitpos = edges[ind].v1 + dir * (val - (ind ? ecdf[ind - 1] : 0));
        output.normal = glm::vec3(dir.y, -dir.x, 0);
    }

    static __forceinline__ __device__ void SampleEdge(Edge* edges, unsigned int nedge, float* ecdf, HitInfo& output, unsigned int removed, curandState& state) {
        float val = curand_uniform(&state) * (ecdf[nedge - 1] - ecdf[removed] + (removed != 0 ? ecdf[removed - 1] : 0));
        if (removed == 0 || val > ecdf[removed - 1]) val += ecdf[removed] - (removed ? ecdf[removed - 1] : 0);
        unsigned int ind = lower_bound(ecdf, nedge, val);
        if (ind >= nedge) ind = nedge - 1;
        glm::vec3 dir = glm::normalize(edges[ind].v2 - edges[ind].v1);
        output.hitpos = edges[ind].v1 + dir * (val - (ind ? ecdf[ind - 1] : 0));
        output.normal = glm::vec3(dir.y, -dir.x, 0);
    }

    static __forceinline__ __device__ float distance(glm::vec3 pos, glm::vec3 v1, glm::vec3 v2) {
        glm::vec3 ed = v2 - v1;
        glm::vec3 norm = glm::normalize(glm::vec3(ed.y, -ed.x, 0));
        float t = glm::dot(v1 - pos, norm);
        glm::vec3 hitpos = pos + t * norm;
        if (glm::dot(v1 - hitpos, v2 - hitpos) <= 0) return t;
        else if (glm::length(v1 - hitpos) < glm::length(v2 - hitpos)) return glm::length(v1 - pos);
        else return glm::length(v2 - pos);
    }

    __forceinline__ __global__ void wob(
        float* nu,
        Edge* edges, unsigned int nedge,
        float* ecdf,
        glm::vec3* nxvorts, curandState* states,
        float t) {
        unsigned int const ind = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int edgeid =
            lower_bound(ecdf, nedge, ecdf[nedge - 1] * ind / (nsamples - 1));
        if (edgeid >= nedge) edgeid = nedge - 1;

        glm::vec3 pos, norm;
        {
            unsigned int const begsample = // The first sample on this edge
                (edgeid ? floor((nsamples - 1) * ecdf[edgeid - 1] / ecdf[nedge - 1]) : 0);

            unsigned int const assignedsample = // The number of samples on this edge
                floor((nsamples - 1) * ecdf[edgeid] / ecdf[nedge - 1]) - begsample + 1;
        
            float t = (ind - begsample + .5f) / assignedsample;
            
            glm::vec3 dir = edges[edgeid].v2 - edges[edgeid].v1;

            pos = edges[edgeid].v1 + t * dir;
            
            norm = glm::normalize(glm::vec3(dir.y, -dir.x, 0));
        }

        curandState& state = states[ind];

        constexpr unsigned int nwob = 4; // The number of times do walk on boundary
        constexpr float RRfactor = 0.8f; // The Russian-Roulette constant
        constexpr unsigned int max_depth = 2u; // The maximum depth of wob

        Status buf[buffersize / sizeof(Status)]; // Buffer, just buffers
        StatStack ss{buf};

        float ans = 0;

        for (unsigned int time = 0; time < nwob; ++time) {
			ss.push_back(Status{ 0,
				1.f,
				glm::vec3(0, 0, 1),
				pos,
				norm });

            while (!ss.empty()) {
                unsigned int const index = ss.sz - 1;
                Status& st = ss.back();
                if (st.normal.z < 2) {
                    // Not bounced
                    if (st.depth >= max_depth || st.depth != 0 && curand_uniform(&state) > RRfactor) {
                        // Terminate here
                        float val = 0;
                        GetVal(st.pos, t, nxvorts, st.normal, val);
                        ans += st.coef * val;
                    }
                    else { // Can keep bouncing
                        // First compute the contribution
                        if (st.depth) {
                            float val = 0;
                            GetVal(st.pos, t, nxvorts, st.normal, val);
                            ans += st.coef * val * (1 + 1 / RRfactor);
                        }
                       
                        //int cnt = 0;
                        // Seek for the first bouncing point
                        while (true) {
                            //if (cnt++ > 10000) {
                                //printf("Cannot find next bouncing point: %f %f %f\n", st.pos.x, st.pos.y, st.pos.z);
                            //}
                            // Loop to make sure hitting the boundary correct times
                            glm::vec3 const dir = st.depth != 0 ? RanddirH2D(st.normal, state) : Randdir2D(state);
                            if (!insidetest(st.pos, dir, edges, nedge)) {
                                //printf("%f %f %f !-> %f %f %f\n", st.pos.x, st.pos.y, st.pos.z, dir.x, dir.y, dir.z);
                                continue;
                            }
                            HitInfo hi;
                            hi.hitdist = 1e10f;
                            TestHit(edges, nedge, Ray{ st.pos, dir }, hi);
                            if (false && st.depth == 0) {
                               glm::vec3 diff = st.pos - hi.hitpos;
                                printf("%f %f %f: %f %f %f\n", diff.x, diff.y, diff.z,
                                    p2Gpxkpny(-st.pos + hi.hitpos, -hi.normal, 0),
                                    p2Gpxkpny(-st.pos + hi.hitpos, -hi.normal, 1),
                                    pGpny(-st.pos + hi.hitpos, -hi.normal));
                            }
                            ss.push_back(Status{
                                st.depth + 1,
                                st.depth != 0 ? -st.coef / RRfactor
                                              : 1.f,
                                dir, hi.hitpos, hi.normal
                            });
                            break;
                        }
                    }
                    ss.Getcontent(index).normal.z = 3;
                    // Mark as bounced
                }
                else {
                    ss.pop_back();

                    if (st.depth > 0) {
                        // Seek for the next hit point

                        HitInfo hi;
                        hi.hitdist = 1e10f;
                        TestHit(edges, nedge, Ray{ st.pos, st.dir }, hi);
                        if (hi.hitdist < 5e9f) {
                            // A hit
                            ss.push_back(Status{
                                st.depth,
                                -st.coef,
                                st.dir, hi.hitpos, hi.normal
                            });
                        }
                    }
                }
            }
        }
        {
            float val = 0;
            GetVal(pos, t, nxvorts, norm, val);
            nu[ind] = 2 * val - 2 * ans / nwob;
        }
        // End of computation
    }

    static __forceinline__ __device__ float Simpson(glm::vec3 const pos, float const lt, float const rt, Edge const ed, glm::vec3 const norm, int const dim) {
        float md = (lt + rt) / 2;
        float lmd = (lt + md) / 2, rmd = (md + rt) / 2;
        glm::vec3 lp = ed.v1 * (1 - lt) + ed.v2 * lt, rp = ed.v1 * (1 - rt) + ed.v2 * rt, mp = ed.v1 * (1 - md) + ed.v2 * md;
        glm::vec3 llp = ed.v1 * (1 - lmd) + ed.v2 * lmd, rrp = ed.v1 * (1 - rmd) + ed.v2 * rmd;
        float val = 
            (p2Gpxkpny(pos - lp, norm, dim) + 4 * p2Gpxkpny(pos - mp, norm, dim) + p2Gpxkpny(pos - rp, norm, dim)) / 6;
        float lval =
            (p2Gpxkpny(pos - lp, norm, dim) + 4 * p2Gpxkpny(pos - llp, norm, dim) + p2Gpxkpny(pos - mp, norm, dim)) / 6;
        float rval =
            (p2Gpxkpny(pos - md, norm, dim) + 4 * p2Gpxkpny(pos - rrp, norm, dim) + p2Gpxkpny(pos - rp, norm, dim)) / 6;
        if (fabs(val * 2 - lval - rval) < 2e-3f) return lval + rval;
        //printf("%f %f: %f\n", lt, rt, val * 2 - lval - rval);
        return Simpson(pos, lt, md, ed, norm, dim) + Simpson(pos, md, rt, ed, norm, dim);
    }

    // Do manual integration
    __forceinline__ __global__ void integrate(float* nu, Edge* edges, unsigned int nedge, float* ecdf, glm::vec3* pV0, glm::vec3* output, curandState* states) {
        unsigned int const ind = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int const y = ind / width, x = ind % width;
        curandState& state = states[ind];
        glm::vec3 const pos = glm::vec3((x + 0.5f) / width, (y + 0.5f) / width, 0);
        {
            // Inside-outside test to expell all points that is outside our focusing area
            unsigned int success = 0u;
            for (unsigned int i = 0; i < 7; ++i)
                if (isInside(pos, edges, nedge, state)) {
                    success++;
                }

            if (success < 5u) {
                // Does not pass the test, set Phi = V0
                //if(pos.x < 0.4f || pos.x > 0.6f || pos.y < 0.4f || pos.y > 0.6f)
                //    printf("You shall pass: %f %f %f -> %u\n", pos.x, pos.y, pos.z, success);
                output[ind].x = pV0[ind].x;
                output[ind].y = pV0[ind].y;
                return;
            }
        }

        float ansx = 0, ansy = 0;

        for (unsigned int edgeid = 0; edgeid < nedge; ++edgeid) {
            unsigned int const begsample = // The first sample on this edge
                (edgeid ? floor((nsamples - 1) * ecdf[edgeid - 1] / ecdf[nedge - 1]) : 0);

            unsigned int const assignedsample = // The number of samples on this edge
                floor((nsamples - 1) * ecdf[edgeid] / ecdf[nedge - 1]) - begsample + 1;
		    
            glm::vec3 const dir = edges[edgeid].v2 - edges[edgeid].v1;

            glm::vec3 const norm = glm::normalize(glm::vec3(dir.y, -dir.x, 0));

            for (unsigned int sampid = 0; sampid < assignedsample; ++sampid) {
			    float const lt = (1.f * sampid - begsample) / assignedsample;
                float const rt = (1.f * sampid - begsample + 1) / assignedsample;
                {
                    float t = (lt + rt) / 2;
                    ansx -= Simpson(pos, lt, rt, edges[edgeid], -norm, 0) * nu[edgeid] * (rt - lt) * glm::length(dir);
                    ansy -= Simpson(pos, lt, rt, edges[edgeid], -norm, 1) * nu[edgeid] * (rt - lt) * glm::length(dir);
                }
            }
        }
        output[ind].x = ansx;
        output[ind].y = ansy;
    }
}
#endif // USE_2D
