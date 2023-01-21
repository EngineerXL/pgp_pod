#ifndef POLYGON_HPP
#define POLYGON_HPP

#include "textures.cuh"

struct ray3d_t {
    vec3d p, v;
    int pix_id;
    vec3f coef;

    __host__ __device__ ray3d_t() : p(), v(), pix_id(), coef() {}

    __host__ __device__ ray3d_t(const vec3d& _p, const vec3d& _v, int _pix_id)
        : p(_p), v(_v), pix_id(_pix_id), coef(1, 1, 1) {
        v.norm();
    }

    __host__ __device__ ray3d_t(const vec3d& _p, const vec3d& _v, int _pix_id,
                                const vec3f& _coef)
        : p(_p), v(_v), pix_id(_pix_id), coef(_coef) {
        v.norm();
    }
};

using ray = ray3d_t;

struct triangle3d_t {
    vec3d a, b, c, n, e1, e2;

    triangle3d_t() = delete;

    __host__ __device__ void init_triangle() {
        n = vec3d::cross(b - a, c - a);
        n.norm();
        e1 = b - a;
        e2 = c - a;
    }

    __host__ __device__ triangle3d_t(const vec3d& _a, const vec3d& _b,
                                     const vec3d& _c)
        : a(_a), b(_b), c(_c) {
        init_triangle();
    }

    __host__ __device__ void shift(const vec3d& v) {
        a += v;
        b += v;
        c += v;
    }

    friend std::ostream& operator<<(std::ostream& out,
                                    const triangle3d_t& trig) {
        out << '{';
        out << "a = " << trig.a;
        out << ", b = " << trig.b;
        out << ", c = " << trig.c;
        out << '}';
        return out;
    }

    void check_norm(const vec3d& _n) {
        if (vec3d::dot(_n, n) < -EPS) {
            std::swap(a, c);
            init_triangle();
        }
    }
};

using triangle = triangle3d_t;

#define EDGE_LIGHT_I 16.0f
#define LIGHT_SIZE 0.025
#define MAGIC 1e-3

struct polygon3d_t {
    triangle trig;
    vec3f color;
    double a, b, c, d;
    float coef_reflection, coef_transparent, coef_blend;
    int n_lights;
    char textured;
    vec3d v1, v2, v3;
    texture_t tex;

    polygon3d_t() = delete;

    __host__ __device__ void build_plane() {
        vec3d p0 = trig.a;
        vec3d v1 = trig.b - p0;
        vec3d v2 = trig.c - p0;
        a = v1.y * v2.z - v1.z * v2.y;
        b = (-1.0) * (v1.x * v2.z - v1.z * v2.x);
        c = v1.x * v2.y - v1.y * v2.x;
        d = -p0.x * (v1.y * v2.z - v1.z * v2.y) +
            (p0.y) * (v1.x * v2.z - v1.z * v2.x) +
            (-p0.z) * (v1.x * v2.y - v1.y * v2.x);
    }

    /* Glass */
    __host__ __device__ polygon3d_t(const triangle& _trig, const vec3f& _color,
                                    float _coef_reflection,
                                    float _coef_transparent)
        : trig(_trig),
          color(_color),
          coef_reflection(_coef_reflection),
          coef_transparent(_coef_transparent),
          coef_blend(1.0f - _coef_reflection - _coef_transparent),
          n_lights(),
          textured(),
          tex() {
        assert(coef_blend > -EPS);
        assert(coef_blend < 1 + EPS);
        build_plane();
    }

    /* Corner */
    __host__ __device__ polygon3d_t(const triangle& _trig, const vec3f& _color)
        : trig(_trig),
          color(_color),
          coef_reflection(),
          coef_transparent(),
          coef_blend(1.0f),
          n_lights(),
          textured(),
          tex() {
        build_plane();
    }

    /* Edge */
    __host__ __device__ polygon3d_t(const triangle& _trig, const vec3f& _color,
                                    int _n_lights, const vec3d& _v1,
                                    const vec3d& _v2)
        : trig(_trig),
          color(_color),
          coef_reflection(),
          coef_transparent(),
          coef_blend(1.0f),
          n_lights(_n_lights),
          textured(),
          v1(_v1),
          v2(_v2),
          tex() {
        assert(n_lights > 0);
        build_plane();
    }

    /* Textured */
    __host__ __device__ polygon3d_t(const triangle& _trig, const vec3f& _color,
                                    float _coef_reflection,
                                    float _coef_transparent, const vec3d& _v1,
                                    const vec3d& _v2, const vec3d& _v3,
                                    const texture_t& _tex)
        : trig(_trig),
          color(_color),
          coef_reflection(_coef_reflection),
          coef_transparent(_coef_transparent),
          coef_blend(1.0f - _coef_reflection - _coef_transparent),
          n_lights(),
          textured(true),
          v1(_v1),
          v2(_v2),
          v3(_v3),
          tex(_tex) {
        assert(coef_blend > -EPS);
        assert(coef_blend < 1 + EPS);
        build_plane();
    }

    __host__ __device__ vec3f get_color(const ray& r, const vec3d& hit) const {
        if (textured) {
            vec3d p = hit - v3;
            double beta =
                (p.x * v1.y - p.y * v1.x) / (v2.x * v1.y - v2.y * v1.x);
            double alpha =
                (p.x * v2.y - p.y * v2.x) / (v1.x * v2.y - v1.y * v2.x);
            return tex.get_pix(alpha, beta);
        } else if (n_lights > 0 && vec3d::dot(trig.n, r.v) > 0.0) {
            vec3d vl = (v2 - v1) / (n_lights + 1);
            for (int i = 1; i <= n_lights; ++i) {
                vec3d p_light = v1 + i * vl;
                if ((p_light - hit).len() < LIGHT_SIZE) {
                    return vec3f(EDGE_LIGHT_I, EDGE_LIGHT_I, EDGE_LIGHT_I);
                }
            }
        }
        return color;
    }

    friend std::ostream& operator<<(std::ostream& out,
                                    const polygon3d_t& poly) {
        out << '{';
        out << "trig = " << poly.trig;
        out << ", color = " << poly.color
            << ", coef_reflection = " << poly.coef_reflection
            << ", coef_transparent = " << poly.coef_transparent;
        out << '}';
        return out;
    }
};

using polygon = polygon3d_t;

__host__ __device__ void intersect_ray_plane(const ray& r, const polygon& poly,
                                             double& _t) {
    _t = -(poly.a * r.p.x + poly.b * r.p.y + poly.c * r.p.z + poly.d) /
         (poly.a * r.v.x + poly.b * r.v.y + poly.c * r.v.z);
}

__host__ __device__ void intersect_ray_polygon(const ray& r,
                                               const polygon& poly, double& _t,
                                               bool& ans) {
    ans = false;
    vec3d P = vec3d::cross(r.v, poly.trig.e2);
    double div = vec3d::dot(P, poly.trig.e1);
    if (fabs(div) < EPS) {
        return;
    }
    vec3d T = r.p - poly.trig.a;
    double u = vec3d::dot(P, T) / div;
    if (u < 0.0 || u > 1.0) {
        return;
    }
    vec3d Q = vec3d::cross(T, poly.trig.e1);
    double v = vec3d::dot(Q, r.v) / div;
    if (v < 0.0 || u + v > 1.0) {
        return;
    }
    _t = vec3d::dot(Q, poly.trig.e2) / div;
    ans = (_t < 0.0 ? false : true);
}

#endif /* POLYGON_HPP */
