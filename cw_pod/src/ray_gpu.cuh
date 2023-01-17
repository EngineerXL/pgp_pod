#ifndef RAY_GPU_HPP
#define RAY_GPU_HPP

#include "scene.cuh"

__global__ void clear_data_gpu(vec3f* dev_data, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += offset) {
        dev_data[i] = vec3f(0, 0, 0);
    }
}

__global__ void trace_gpu(const ray* rays_in, const int sz_in, ray* rays_out,
                          int* sz_out, vec3f* dev_data,
                          const light_source_t* dev_lights, int n_sources,
                          const polygon* dev_polys, int n_polygons) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int k = idx; k < sz_in; k += offset) {
        int min_i = n_polygons;
        double min_t = INF;
        for (int i = 0; i < n_polygons; i++) {
            double t;
            bool flag;
            intersect_ray_polygon(rays_in[k], dev_polys[i], t, flag);
            if (flag && t < min_t) {
                min_i = i;
                min_t = t;
            }
        }
        if (min_i == n_polygons) {
            continue;
        }
        vec3d hit = rays_in[k].p + min_t * rays_in[k].v;
        vec3f poly_color = dev_polys[min_i].get_color(rays_in[k], hit);
        vec3f::atomicAdd_vec(
            &dev_data[rays_in[k].pix_id],
            phong_shading(rays_in[k], hit, dev_polys[min_i], min_i, dev_lights,
                          n_sources, dev_polys, n_polygons));
        if (dev_polys[min_i].coef_transparent > 0) {
            rays_out[atomicAdd(sz_out, 1)] =
                ray(hit + MAGIC * rays_in[k].v, rays_in[k].v, rays_in[k].pix_id,
                    dev_polys[min_i].coef_transparent * rays_in[k].coef *
                        poly_color);
        }
        if (dev_polys[min_i].coef_reflection > 0) {
            vec3d reflected =
                vec3d::reflect(rays_in[k].v, dev_polys[min_i].trig.n);
            rays_out[atomicAdd(sz_out, 1)] =
                ray(hit + MAGIC * reflected, reflected, rays_in[k].pix_id,
                    dev_polys[min_i].coef_reflection * rays_in[k].coef *
                        poly_color);
        }
    }
}

__global__ void init_rays_gpu(const vec3d pc, const vec3d pv, int w, int h,
                              double angle, ray* dev_rays) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / std::tan(angle * M_PI / 360.0);
    vec3d bz = pv - pc;
    vec3d bx = vec3d::cross(bz, vec3d(0, 0, 1));
    vec3d by = vec3d::cross(bx, bz);
    bx.norm();
    by.norm();
    bz.norm();

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            vec3d v(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3d dir = mult(bx, by, bz, v);
            int pix_id = (h - 1 - j) * w + i;
            dev_rays[i * h + j] = ray(pc, dir, pix_id);
        }
    }
}

__global__ void write_data_gpu(uchar4* dev_data, vec3f* dev_data_vec3f,
                               int sz) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < sz; i += offset) {
        dev_data_vec3f[i].clamp();
        dev_data_vec3f[i] *= 255.0f;
        dev_data[i] = make_uchar4(dev_data_vec3f[i].x, dev_data_vec3f[i].y,
                                  dev_data_vec3f[i].z, 255);
    }
}

void render_gpu(int frame_id, const vec3d pc, const vec3d pv, int w, int h,
                double angle, uchar4* dev_data) {
    int sz_in = w * h;
    vec3f* dev_data_vec3f;
    CSC(cudaMalloc(&dev_data_vec3f, sizeof(vec3f) * sz_in));
    clear_data_gpu<<<BLOCKS, THREADS>>>(dev_data_vec3f, w * h);
    ray* dev_ray_in;
    CSC(cudaMalloc(&dev_ray_in, sizeof(ray) * sz_in));
    init_rays_gpu<<<BLOCKS_2D, THREADS_2D>>>(pc, pv, w, h, angle, dev_ray_in);
    CSC(cudaGetLastError());
    const int ZERO = 0;
    long long total_rays = 0;
    cuda_timer_t tmr;
    tmr.start();
    for (int rec = 0; rec < rec_depth and sz_in; ++rec) {
        total_rays += sz_in;
        ray* dev_ray_out;
        CSC(cudaMalloc(&dev_ray_out, 2 * sizeof(ray) * sz_in));
        int* sz_out;
        CSC(cudaMalloc(&sz_out, sizeof(int)));
        CSC(cudaMemcpy(sz_out, &ZERO, sizeof(int), cudaMemcpyHostToDevice));
        trace_gpu<<<BLOCKS, THREADS>>>(dev_ray_in, sz_in, dev_ray_out, sz_out,
                                       dev_data_vec3f, dev_lights, n_sources,
                                       dev_polys, n_polygons);
        CSC(cudaGetLastError());
        CSC(cudaFree(dev_ray_in));
        dev_ray_in = dev_ray_out;
        CSC(cudaMemcpy(&sz_in, sz_out, sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaFree(sz_out));
        CSC(cudaGetLastError());
    }
    write_data_gpu<<<BLOCKS, THREADS>>>(dev_data, dev_data_vec3f, w * h);
    tmr.end();
    CSC(cudaFree(dev_ray_in));
    CSC(cudaFree(dev_data_vec3f));
    CSC(cudaGetLastError());
    printf("%d\t%.3lf\t%lli\n", frame_id, tmr.get_time(), total_rays);
    fflush(stdout);
}

#endif /* RAY_GPU_HPP */
