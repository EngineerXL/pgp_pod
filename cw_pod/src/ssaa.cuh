#ifndef SSAA_HPP
#define SSAA_HPP

#define to_ind(i, j, dim) ((j) * (dim) + (i))

void ssaa_cpu(const uchar4 *data_in, uchar4 *data_out, int w, int h, int coef) {
    double coef2 = coef * coef;
    // int hc = h * coef;
    int wc = w * coef;
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            double r = 0, g = 0, b = 0;
            for (int ki = 0; ki < coef; ++ki) {
                for (int kj = 0; kj < coef; ++kj) {
                    uchar4 pix =
                        data_in[to_ind(coef * i + ki, coef * j + kj, wc)];
                    r += pix.x;
                    g += pix.y;
                    b += pix.z;
                }
            }
            r /= coef2;
            g /= coef2;
            b /= coef2;
            data_out[to_ind(i, j, w)] = make_uchar4(r, g, b, 255);
        }
    }
}

__global__ void ssaa_gpu(const uchar4 *data_in, uchar4 *data_out, int w, int h,
                         int coef) {
    double coef2 = coef * coef;
    // int hc = h * coef;
    int wc = w * coef;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            double r = 0, g = 0, b = 0;
            for (int ki = 0; ki < coef; ++ki) {
                for (int kj = 0; kj < coef; ++kj) {
                    uchar4 pix =
                        data_in[to_ind(coef * i + ki, coef * j + kj, wc)];
                    r += pix.x;
                    g += pix.y;
                    b += pix.z;
                }
            }
            r /= coef2;
            g /= coef2;
            b /= coef2;
            data_out[to_ind(i, j, w)] = make_uchar4(r, g, b, 255);
        }
    }
}

#endif /* SSAA_HPP */
