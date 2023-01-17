#ifndef SCENE_HPP
#define SCENE_HPP

#include <algorithm>
#include <fstream>
#include <set>

#include "variables_gpu.cuh"

const double EDGE_THICKNESS = 0.1;
const vec3f EDGE_COLOR(0.25f, 0.25f, 0.25f);

void read_obj(int id, std::string fname) {
    std::ifstream f(fname);
    std::string s;
    std::vector<vec3d> vertices;
    std::vector<std::set<int> > vertex_polygons;
    std::vector<vec3i> polygons;
    int polygon_id = 0;
    while (f >> s) {
        if (s == "v") {
            vec3d vertex;
            f >> vertex;
            vertex *= figs[id].radius;
            vertices.push_back(vertex);
            vertex_polygons.push_back(std::set<int>());
        } else if (s == "f") {
            vec3i ids;
            f >> ids;
            --ids.x;
            --ids.y;
            --ids.z;
            polygons.push_back(ids);
            vertex_polygons[ids.x].insert(polygon_id);
            vertex_polygons[ids.y].insert(polygon_id);
            vertex_polygons[ids.z].insert(polygon_id);
            ++polygon_id;
        }
    }
    f.close();
    double side = INF;
    int m = vertices.size();
    for (int i = 0; i < m; ++i) {
        vec3d vi = vertices[i];
        for (int j = i + 1; j < m; ++j) {
            vec3d vj = vertices[j];
            side = std::min(side, (vi - vj).len());
        }
    }
    std::set<int> unique_polygon_ids;
    for (int i = 0; i < m; ++i) {
        vec3d vi = vertices[i];
        for (int j = i + 1; j < m; ++j) {
            vec3d vj = vertices[j];
            if ((vi - vj).len() > side + EPS) {
                continue;
            }
            std::vector<int> trig_ids;
            std::vector<triangle> trigs;
            for (int elem : vertex_polygons[i]) {
                if (vertex_polygons[j].count(elem)) {
                    trig_ids.push_back(elem);
                    vec3i ids = polygons[elem];
                    trigs.push_back(triangle(vertices[ids.x], vertices[ids.y],
                                             vertices[ids.z]));
                }
            }
            assert(trigs.size() == 2);
            double t;
            bool unused_flag;
            int id1 = trig_ids[0];
            int id2 = trig_ids[1];
            triangle trig1 = trigs[0];
            triangle trig2 = trigs[1];
            vec3d n1 = EDGE_THICKNESS * trig1.n;
            vec3d n2 = EDGE_THICKNESS * trig2.n;
            vec3d n_avg = (n1 + n2) / 2;
            trig1.shift(n1);
            trig2.shift(n2);
            vec3d vi1 = vi + n1;
            vec3d vi2 = vi + n2;
            vec3d vj1 = vj + n1;
            vec3d vj2 = vj + n2;
            vec3d vi_avg = (vi1 + vi2) / 2 + figs[id].center;
            vec3d vj_avg = (vj1 + vj2) / 2 + figs[id].center;

            triangle edge1(vi1, vj2, vi2);
            edge1.check_norm(n_avg);
            intersect_ray_polygon(ray(vec3d(0, 0, 0), vi, -1),
                                  polygon(edge1, EDGE_COLOR), t, unused_flag);
            triangle corneri(vi1, vi2, t * vi / vi.len());
            corneri.check_norm(n_avg);

            edge1.shift(figs[id].center);
            corneri.shift(figs[id].center);
            polys.push_back(
                polygon(edge1, EDGE_COLOR, figs[id].n_lights, vi_avg, vj_avg));
            polys.push_back(polygon(corneri, EDGE_COLOR));

            triangle edge2(vi1, vj1, vj2);
            edge2.check_norm(n_avg);
            intersect_ray_polygon(ray(vec3d(0, 0, 0), vj, -1),
                                  polygon(edge2, EDGE_COLOR), t, unused_flag);
            triangle cornerj(vj1, t * vj / vj.len(), vj2);
            cornerj.check_norm(n_avg);

            edge2.shift(figs[id].center);
            cornerj.shift(figs[id].center);
            polys.push_back(
                polygon(edge2, EDGE_COLOR, figs[id].n_lights, vi_avg, vj_avg));
            polys.push_back(polygon(cornerj, EDGE_COLOR));

            if (!unique_polygon_ids.count(id1)) {
                trig1.shift(figs[id].center);
                polys.push_back(polygon(trig1, figs[id].color,
                                        figs[id].coef_reflection,
                                        figs[id].coef_transparent));
                unique_polygon_ids.insert(id1);
            }
            if (!unique_polygon_ids.count(id2)) {
                trig2.shift(figs[id].center);
                polys.push_back(polygon(trig2, figs[id].color,
                                        figs[id].coef_reflection,
                                        figs[id].coef_transparent));
                unique_polygon_ids.insert(id2);
            }
        }
    }
    assert(unique_polygon_ids.size() == polygons.size());
}

texture_t floor_texture;

void init_scene() {
    floor_texture.load(floor_tex.c_str(), device_gpu);
    triangle t1 = {floor_p[0], floor_p[2], floor_p[1]};
    triangle t2 = {floor_p[2], floor_p[0], floor_p[3]};
    polys.push_back(polygon(t1, floor_color, floor_reflection, 0.0,
                            floor_p[1] - floor_p[2], floor_p[1] - floor_p[0],
                            floor_p[0] + floor_p[2] - floor_p[1],
                            floor_texture));
    polys.push_back(polygon(t2, floor_color, floor_reflection, 0.0,
                            floor_p[0] - floor_p[3], floor_p[2] - floor_p[3],
                            floor_p[3], floor_texture));
    read_obj(0, "objs/cube.obj");
    read_obj(1, "objs/octahedron.obj");
    read_obj(2, "objs/dodecahedron.obj");
    n_polygons = polys.size();
}

__host__ __device__ vec3d mult(const vec3d& a, const vec3d& b, const vec3d& c,
                               const vec3d& v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
}

#define COEF_AMBIENT 0.25
#define COEF_SOURCE 1.0
#define K_D 1.0
#define K_S 0.5

__host__ __device__ vec3f phong_shading(const ray r, const vec3d hit,
                                        const polygon poly, int id,
                                        const light_source_t* lights,
                                        int n_sources, const polygon* polys,
                                        int n_polygons) {
    vec3f poly_color = poly.get_color(r, hit);
    vec3f clr = COEF_AMBIENT * poly.coef_blend * r.coef * poly_color;
    for (int j = 0; j < n_sources; ++j) {
        double t_max = (lights[j].p - hit).len();
        ray r_light(hit, lights[j].p - hit, r.pix_id);
        vec3f coef_vis(1, 1, 1);
        for (int i = 0; i < n_polygons; ++i) {
            if (i == id) {
                continue;
            }
            double t;
            bool flag;
            intersect_ray_polygon(r_light, polys[i], t, flag);
            if (flag and t < t_max) {
                coef_vis *= polys[i].coef_transparent;
            }
        }
        vec3f clr_a =
            poly.coef_blend * r.coef * coef_vis * lights[j].i * poly_color;
        double coef_diffusal = vec3d::dot(poly.trig.n, r_light.v);
        vec3d reflected = vec3d::reflect(r_light.v, poly.trig.n);
        double coef_specular = vec3d::dot(reflected, r.v);
        if (coef_specular < 0.0) {
            coef_specular = 0.0;
        }
        if (coef_diffusal < 0.0) {
            coef_diffusal = 0.0;
            coef_specular = 0.0;
        }
        coef_specular = std::pow(coef_specular, 9);
        clr +=
            COEF_SOURCE * (K_D * coef_diffusal + K_S * coef_specular) * clr_a;
        clr.clamp();
    }
    return clr;
}

#endif /* SCENE_HPP */
