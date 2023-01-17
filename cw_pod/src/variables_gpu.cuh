#ifndef VARS_GPU_HPP
#define VARS_GPU_HPP

#include "variables_cpu.cuh"

const int BLOCKS = 256;
const int THREADS = 256;

const dim3 BLOCKS_2D(64, 64);
const dim3 THREADS_2D(1, 32);

light_source_t* dev_lights;
polygon* dev_polys;

void init_gpu_variables() {
    CSC(cudaMalloc(&dev_lights, sizeof(light_source_t) * n_sources));
    CSC(cudaMemcpy(dev_lights, lights.data(),
                   sizeof(light_source_t) * n_sources, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_polys, sizeof(polygon) * n_polygons));
    CSC(cudaMemcpy(dev_polys, polys.data(), sizeof(polygon) * n_polygons,
                   cudaMemcpyHostToDevice));
}

void destroy_gpu_variables() {
    CSC(cudaFree(dev_lights));
    CSC(cudaFree(dev_polys));
}

#endif /* VARS_GPU_HPP */
