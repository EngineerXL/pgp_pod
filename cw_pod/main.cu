#include <math.h>
#include <stdlib.h>

#include "src/ray_cpu.cuh"
#include "src/ray_gpu.cuh"
#include "src/ssaa.cuh"

using namespace std;

void read_input() {
    cin >> frames;
    cin >> path;
    cin >> w >> h >> angle;
    cin >> r0_c >> z0_c >> phi0_c >> Ar_c >> Az_c >> wr_c >> wz_c >> wphi_c >>
        pr_c >> pz_c;
    cin >> r0_n >> z0_n >> phi0_n >> Ar_n >> Az_n >> wr_n >> wz_n >> wphi_n >>
        pr_n >> pz_n;
    for (int i = 0; i < 3; ++i) {
        cin >> figs[i];
    }
    for (int i = 0; i < N_FLOOR_POINTS; ++i) {
        cin >> floor_p[i];
    }
    cin >> floor_tex >> floor_color >> floor_reflection;
    cin >> n_sources;
    lights.resize(n_sources);
    for (int i = 0; i < n_sources; ++i) {
        cin >> lights[i];
    }
    cin >> rec_depth >> coef_ssaa;
}

void print_file(const char *fname) {
    FILE *in = fopen(fname, "r");
    while (!feof(in)) {
        char c = getc(in);
        if (c == EOF) {
            break;
        }
        printf("%c", c);
    }
    fclose(in);
}

int main(int argc, char *argv[]) {
    assert(argc <= 2);
    device_gpu = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0) {
            device_gpu = 0;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            device_gpu = 1;
        } else if (strcmp(argv[i], "--default") == 0) {
            print_file("default.in");
            return 0;
        } else {
            printf("Unknown key: %s\n", argv[i]);
            return 0;
        }
    }
    read_input();
    init_scene();
    init_gpu_variables();
    double tau = 2 * M_PI / frames;
    int w_ssaa = w * coef_ssaa;
    int h_ssaa = h * coef_ssaa;
    char buff[256];
    uchar4 *data_ssaa = new uchar4[w_ssaa * h_ssaa];
    uchar4 *data = new uchar4[w * h];
    uchar4 *dev_data_ssaa;
    CSC(cudaMalloc(&dev_data_ssaa, sizeof(uchar4) * w_ssaa * h_ssaa));
    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    for (int k = 0; k < frames; k++) {
        vec3d pc, pv;
        double t = k * tau;
        pc = vec3d::from_cyl(r0_c + Ar_c * sin(wr_c * t + pr_c),
                             z0_c + Az_c * sin(wz_c * t + pz_c),
                             phi0_c + wphi_c * t);
        pv = vec3d::from_cyl(r0_n + Ar_n * sin(wr_n * t + pr_n),
                             z0_n + Az_n * sin(wz_n * t + pz_n),
                             phi0_n + wphi_n * t);
        if (device_gpu > 0) {
            render_gpu(k, pc, pv, w_ssaa, h_ssaa, angle, dev_data_ssaa);
            ssaa_gpu<<<BLOCKS_2D, THREADS_2D>>>(dev_data_ssaa, dev_data, w, h,
                                                coef_ssaa);
            CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h,
                           cudaMemcpyDeviceToHost));
        } else {
            render_cpu(k, pc, pv, w_ssaa, h_ssaa, angle, data_ssaa);
            ssaa_cpu(data_ssaa, data, w, h, coef_ssaa);
        }
        sprintf(buff, path.c_str(), k);
        write_output(buff, w, h, data);
    }
    CSC(cudaFree(dev_data_ssaa));
    CSC(cudaFree(dev_data));
    free(data);
    destroy_gpu_variables();
    CSC(cudaGetLastError());
    return 0;
}
