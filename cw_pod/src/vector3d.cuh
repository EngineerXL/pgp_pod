#ifndef VECTOR3D_CUH
#define VECTOR3D_CUH

#include <cmath>
#include <iostream>

const double EPS = 1e-6;
const double INF = 1e18;

template <class T>
struct vector3d_t {
    T x, y, z;

    using vec3d = vector3d_t<T>;

    __host__ __device__ vector3d_t() : x(), y(), z(){};

    __host__ __device__ vector3d_t(const T& _x, const T& _y, const T& _z)
        : x(_x), y(_y), z(_z){};

    __host__ __device__ vector3d_t(const vec3d& v) : x(v.x), y(v.y), z(v.z){};

    __host__ __device__ friend vec3d operator+(const vec3d& a, const vec3d& b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    __host__ __device__ vec3d& operator+=(const vec3d& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ friend vec3d operator-(const vec3d& a, const vec3d& b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    __host__ __device__ vec3d& operator-=(const vec3d& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __host__ __device__ friend vec3d operator*(const vec3d& a, const vec3d& b) {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }

    __host__ __device__ vec3d& operator*=(const vec3d& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ friend vec3d operator*(const T& lambda,
                                               const vec3d& v) {
        return {lambda * v.x, lambda * v.y, lambda * v.z};
    }

    __host__ __device__ vec3d& operator*=(const T& lambda) {
        x *= lambda;
        y *= lambda;
        z *= lambda;
        return *this;
    }

    __host__ __device__ friend vec3d operator/(const vec3d& v,
                                               const T& lambda) {
        return {v.x / lambda, v.y / lambda, v.z / lambda};
    }

    __host__ __device__ vec3d& operator/=(const T& lambda) {
        x /= lambda;
        y /= lambda;
        z /= lambda;
        return *this;
    }

    __host__ __device__ static vec3d from_cyl(const T& _r, const T& _z,
                                              const T& _phi) {
        return {_r * std::cos(_phi), _r * std::sin(_phi), _z};
    };

    friend std::istream& operator>>(std::istream& in, vec3d& v) {
        in >> v.x >> v.y >> v.z;
        return in;
    }

    friend std::ostream& operator<<(std::ostream& out, const vec3d& v) {
        // out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        out << v.x << ", " << v.y << ", " << v.z;
        return out;
    }

    __host__ __device__ static double dot(const vec3d& a, const vec3d& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ double len() const {
        return std::sqrt(dot(*this, *this));
    }

    __host__ __device__ void norm() {
        double l = std::sqrt(dot(*this, *this));
        x /= l;
        y /= l;
        z /= l;
    }

    __host__ __device__ static vec3d cross(const vec3d& a, const vec3d& b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
    }

    __host__ __device__ static double square(const vec3d& a, const vec3d& b) {
        return cross(a, b).len();
    }

    __host__ __device__ static float clamp(float val) {
        if (val > 1.0f) {
            return 1.0f;
        }
        if (val < 0.0f) {
            return 0.0f;
        }
        return val;
    }

    __host__ __device__ void clamp() {
        x = clamp(x);
        y = clamp(y);
        z = clamp(z);
    }

    /*
     * l   n   r
     * ^   ^   ^
     *  \  |  /
     *   \ | /
     *    \|/
     * ---------
     */
    __host__ __device__ static vec3d reflect(const vec3d& l, const vec3d& n) {
        vec3d r = l - 2 * dot(n, l) * n;
        r.norm();
        return r;
    }

    __device__ static void atomicAdd_vec(vec3d* a, const vec3d& b) {
        atomicAdd(&(a->x), b.x);
        atomicAdd(&(a->y), b.y);
        atomicAdd(&(a->z), b.z);
    }
};

/* Basic types */
using vec3f = vector3d_t<float>;
using vec3d = vector3d_t<double>;
using vec3i = vector3d_t<int>;

#endif /* VECTOR3D_CUH */
