#ifndef TEXTURES_CUH
#define TEXTURES_CUH

#include "io.cuh"
#include "utils.cuh"
#include "vector3d.cuh"

struct texture_t {
    int w, h;
    char device;
    uchar4 *data;
    uchar4 *dev_data;

    __host__ __device__ texture_t()
        : w(0), h(0), device(false), data(nullptr), dev_data(nullptr) {}

    void load(const char *fname, char _device) {
        device = _device;
        data = read_input(fname, w, h);
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
        CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h,
                       cudaMemcpyHostToDevice));
    }

    __host__ __device__ texture_t(const texture_t &other)
        : w(other.w), h(other.h), device(other.device) {
        data = other.data;
        dev_data = other.dev_data;
    }

    __host__ __device__ vec3f get_pix(double x, double y) const {
        int xp = x * w;
        int yp = y * h;
        xp = max(0, min(xp, w - 1));
        yp = max(0, min(yp, h - 1));
        uchar4 p;
        if (device) {
            p = dev_data[yp * w + xp];
        } else {
            p = data[yp * w + xp];
        }
        vec3f res(p.x, p.y, p.z);
        res /= 255.0f;
        return res;
    }
};

#endif /* TEXTURES_CUH */
