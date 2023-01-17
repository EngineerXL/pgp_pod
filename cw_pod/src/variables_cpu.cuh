#ifndef VARS_CPU_HPP
#define VARS_CPU_HPP

#include <iostream>
#include <string>
#include <vector>

#include "polygon.cuh"

struct figure_t {
    vec3d center;
    vec3f color;
    double radius;
    double coef_reflection, coef_transparent, n_lights;

    friend std::istream& operator>>(std::istream& in, figure_t& fig) {
        in >> fig.center >> fig.color >> fig.radius >> fig.coef_reflection >>
            fig.coef_transparent >> fig.n_lights;
        return in;
    }
};

struct light_source_t {
    vec3d p;
    vec3f i;

    friend std::istream& operator>>(std::istream& in, light_source_t& source) {
        in >> source.p >> source.i;
        return in;
    }
};

const int N_FIGS = 3;
const int N_FLOOR_POINTS = 4;

char device_gpu;

int n_polygons;
std::vector<polygon> polys;

int frames;
std::string path;
int w, h;
double angle;
double r0_c, z0_c, phi0_c, Ar_c, Az_c, wr_c, wz_c, wphi_c, pr_c, pz_c;
double r0_n, z0_n, phi0_n, Ar_n, Az_n, wr_n, wz_n, wphi_n, pr_n, pz_n;
std::vector<figure_t> figs(N_FIGS);
std::vector<vec3d> floor_p(N_FLOOR_POINTS);
std::string floor_tex;
vec3f floor_color;
double floor_reflection;
int n_sources;
std::vector<light_source_t> lights;
int rec_depth, coef_ssaa;

#endif /* VARS_CPU_HPP */
