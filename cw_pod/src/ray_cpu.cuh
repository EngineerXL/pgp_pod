#ifndef RAY_CPU_HPP
#define RAY_CPU_HPP

#include "scene.cuh"

void clear_data(vec3f* data, const int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = vec3f(0, 0, 0);
    }
}

void trace(const ray* rays_in, int sz_in, ray* rays_out, int& sz_out,
           vec3f* data) {
    for (int k = 0; k < sz_in; ++k) {
        int min_i = n_polygons;
        double min_t = INF;
        for (int i = 0; i < n_polygons; i++) {
            double t;
            bool flag;
            intersect_ray_polygon(rays_in[k], polys[i], t, flag);
            if (flag && t < min_t) {
                min_i = i;
                min_t = t;
            }
        }
        if (min_i == n_polygons) {
            continue;
        }
        vec3d hit = rays_in[k].p + min_t * rays_in[k].v;
        data[rays_in[k].pix_id] +=
            phong_shading(rays_in[k], hit, polys[min_i], min_i, lights.data(),
                          n_sources, polys.data(), n_polygons);
        if (polys[min_i].coef_transparent > 0) {
            rays_out[sz_out++] =
                ray(hit + MAGIC * rays_in[k].v, rays_in[k].v, rays_in[k].pix_id,
                    polys[min_i].coef_transparent * rays_in[k].coef *
                        polys[min_i].get_color(rays_in[k], hit));
        }
        if (polys[min_i].coef_reflection > 0) {
            vec3d reflected = vec3d::reflect(rays_in[k].v, polys[min_i].trig.n);
            rays_out[sz_out++] =
                ray(hit + MAGIC * reflected, reflected, rays_in[k].pix_id,
                    polys[min_i].coef_reflection * rays_in[k].coef *
                        polys[min_i].get_color(rays_in[k], hit));
        }
    }
}

void init_rays(const vec3d& pc, const vec3d& pv, int w, int h, double angle,
               ray* rays) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / std::tan(angle * M_PI / 360.0);
    vec3d bz = pv - pc;
    vec3d bx = vec3d::cross(bz, vec3d(0, 0, 1));
    vec3d by = vec3d::cross(bx, bz);
    bx.norm();
    by.norm();
    bz.norm();
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3d v(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3d dir = mult(bx, by, bz, v);
            int pix_id = (h - 1 - j) * w + i;
            rays[i * h + j] = ray(pc, dir, pix_id);
        }
    }
}

void write_data(uchar4* data, vec3f* data_vec3f, int sz) {
    for (int i = 0; i < sz; ++i) {
        data_vec3f[i].clamp();
        data_vec3f[i] *= 255.0f;
        data[i] =
            make_uchar4(data_vec3f[i].x, data_vec3f[i].y, data_vec3f[i].z, 255);
    }
}

void render_cpu(int frame_id, const vec3d& pc, const vec3d& pv, int w, int h,
                double angle, uchar4* data) {
    int sz_in = w * h;
    vec3f* data_vec3f = new vec3f[sz_in];
    clear_data(data_vec3f, w * h);
    ray* ray_in = new ray[sz_in];
    init_rays(pc, pv, w, h, angle, ray_in);
    long long total_rays = 0;
    cuda_timer_t tmr;
    tmr.start();
    for (int rec = 0; rec < rec_depth and sz_in; ++rec) {
        total_rays += sz_in;
        ray* ray_out = new ray[2 * sz_in];
        int sz_out = 0;
        trace(ray_in, sz_in, ray_out, sz_out, data_vec3f);
        delete[] ray_in;
        ray_in = ray_out;
        sz_in = sz_out;
    }
    write_data(data, data_vec3f, w * h);
    tmr.end();
    delete[] ray_in;
    delete[] data_vec3f;
    printf("%d\t%.3lf\t%lli\n", frame_id, tmr.get_time(), total_rays);
}

#endif /* RAY_CPU_HPP */
