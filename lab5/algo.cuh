#ifndef ALGO_HPP
#define ALGO_HPP

#include <utility>
#include <vector>

#include "../cuda_utils.cuh"

/* Computes ceil lg_2(x) */
int lg_2(int64_t x) {
    int y = 0;
    while ((1 << y) < x) {
        ++y;
    }
    return y;
}

const int GLOBAL_BLOCKS = 64;
const int GLOBAL_THREADS = 256;

/*
 * Shared memory warning!
 * In template kernel we should
 * cast pointer to T*
 * https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
 */

#ifndef MIN_MAX_REDUCTION
#define MIN_MAX_REDUCTION

const int RED_BLOCKS = GLOBAL_BLOCKS;
const int RED_THREADS = GLOBAL_THREADS;

/*
 * This section contains parallel reduction min(T*) and max(T*)
 * Note, that T* must be device ptr
 */

template <class T>
__global__ void max_val_reduction(const T* gl_a_in, int gl_n, T* gl_a_out) {
    extern __shared__ char sh_mem[];
    T* sh_a = (T*)sh_mem;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsz = blockDim.x;
    int offset = 2 * bid * bsz;
    while (offset + 2 * tid + 1 < gl_n) {
        /* Load data into shared memory */
        sh_a[tid] =
            cuda_max(gl_a_in[offset + 2 * tid], gl_a_in[offset + 2 * tid + 1]);
        __syncthreads();

        /* It could be less elements, then count of blocks */
        int elems_left = (gl_n - offset) / 2;
        int step = cuda_min(bsz, elems_left);
        step >>= 1;
        while (step) {
            if (tid < step) {
                sh_a[tid] = cuda_max(sh_a[tid], sh_a[tid + step]);
            }
            step >>= 1;
            __syncthreads();
        }
        if (tid == 0) {
            gl_a_out[bid] = sh_a[0];
        }
        offset += 2 * bsz * RED_BLOCKS;
        bid += RED_BLOCKS;
        __syncthreads();
    }
}

template <class T>
T max_val(const T* dev_a, int n) {
    int lg_n = lg_2(n);
    int m = 1 << lg_n;

    /* Make copy of size equals to the power of 2 */
    T* dev_a_pad;
    CSC(cudaMalloc(&dev_a_pad, sizeof(T) * m));
    CSC(cudaMemcpy(dev_a_pad, dev_a, sizeof(T) * n, cudaMemcpyDeviceToDevice));
    /* Pad array */
    CSC(cudaMemcpy(dev_a_pad + n, dev_a, sizeof(T) * (m - n),
                   cudaMemcpyDeviceToDevice));
    CSC(cudaGetLastError());

    /* dev_a_in -----|reduction|-----> dev_a_out */
    T* dev_a_in = dev_a_pad;
    T* dev_a_out;
    int out_sz = std::max(1, m / (2 * RED_THREADS));
    CSC(cudaMalloc(&dev_a_out, sizeof(T) * out_sz));
    CSC(cudaGetLastError());

    /* Each iteration reduces array size by THREADS_PER_BLOCK times */
    int sh_mem = sizeof(T) * RED_THREADS;
    while (m > 2 * RED_THREADS) {
        max_val_reduction<<<RED_BLOCKS, RED_THREADS, sh_mem>>>(dev_a_in, m,
                                                               dev_a_out);
        CSC(cudaGetLastError());
        std::swap(dev_a_in, dev_a_out);
        m /= (2 * RED_THREADS);
    }

    /* Last step --- compute answer in a single block */
    max_val_reduction<<<RED_BLOCKS, RED_THREADS, sh_mem>>>(dev_a_in, m,
                                                           dev_a_out);
    CSC(cudaGetLastError());

    T ans;
    CSC(cudaMemcpy(&ans, dev_a_out, sizeof(T) * 1, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_a_in));
    CSC(cudaFree(dev_a_out));
    CSC(cudaGetLastError());
    return ans;
}

template <class T>
__global__ void min_val_reduction(const T* gl_a_in, int gl_n, T* gl_a_out) {
    extern __shared__ char sh_mem[];
    T* sh_a = (T*)sh_mem;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsz = blockDim.x;
    int offset = 2 * bid * bsz;
    while (offset + 2 * tid + 1 < gl_n) {
        /* Load data into shared memory */
        sh_a[tid] =
            cuda_min(gl_a_in[offset + 2 * tid], gl_a_in[offset + 2 * tid + 1]);
        __syncthreads();

        /* It could be less elements, then count of blocks */
        int elems_left = (gl_n - offset) / 2;
        int step = cuda_min(bsz, elems_left);
        step >>= 1;
        while (step) {
            if (tid < step) {
                sh_a[tid] = cuda_min(sh_a[tid], sh_a[tid + step]);
            }
            step >>= 1;
            __syncthreads();
        }
        if (tid == 0) {
            gl_a_out[bid] = sh_a[0];
        }
        offset += 2 * bsz * RED_BLOCKS;
        bid += RED_BLOCKS;
        __syncthreads();
    }
}

template <class T>
T min_val(const T* dev_a, int n) {
    int lg_n = lg_2(n);
    int m = 1 << lg_n;

    /* Make copy of size equals to the power of 2 */
    T* dev_a_pad;
    CSC(cudaMalloc(&dev_a_pad, sizeof(T) * m));
    CSC(cudaMemcpy(dev_a_pad, dev_a, sizeof(T) * n, cudaMemcpyDeviceToDevice));
    /* Pad array */
    CSC(cudaMemcpy(dev_a_pad + n, dev_a, sizeof(T) * (m - n),
                   cudaMemcpyDeviceToDevice));
    CSC(cudaGetLastError());

    /* dev_a_in -----|reduction|-----> dev_a_out */
    T* dev_a_in = dev_a_pad;
    T* dev_a_out;
    int out_sz = std::max(1, m / (2 * RED_THREADS));
    CSC(cudaMalloc(&dev_a_out, sizeof(T) * out_sz));
    CSC(cudaGetLastError());

    /* Each iteration reduces array size by THREADS_PER_BLOCK times */
    int sh_mem = sizeof(T) * RED_THREADS;
    while (m > 2 * RED_THREADS) {
        min_val_reduction<<<RED_BLOCKS, RED_THREADS, sh_mem>>>(dev_a_in, m,
                                                               dev_a_out);
        CSC(cudaGetLastError());
        std::swap(dev_a_in, dev_a_out);
        m /= (2 * RED_THREADS);
    }

    /* Last step --- compute answer in a single block */
    min_val_reduction<<<RED_BLOCKS, RED_THREADS, sh_mem>>>(dev_a_in, m,
                                                           dev_a_out);
    CSC(cudaGetLastError());

    T ans;
    CSC(cudaMemcpy(&ans, dev_a_out, sizeof(T) * 1, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_a_in));
    CSC(cudaFree(dev_a_out));
    CSC(cudaGetLastError());
    return ans;
}

#endif /* MIN_MAX_REDUCTION */

#ifndef SCAN
#define SCAN

const int SCAN_BLOCKS = GLOBAL_BLOCKS;
const int SCAN_THREADS = GLOBAL_THREADS;

/*
 * This section contains
 * parallel scan algorithm
 */

template <class T>
__device__ void scan_block(T* gl_a, T* gl_sums, T* sh_a, int tid, int bid,
                           int bsz, int pref_sz, int offset) {
    /* Load data into shared memory */
    sh_a[2 * tid] = gl_a[offset + 2 * tid];
    sh_a[2 * tid + 1] = gl_a[offset + 2 * tid + 1];
    __syncthreads();

    /*
     * step --- offset in scan
     * alive --- number of live threads in block
     */
    int step, alive;

    /* leaves ------> root */
    step = 1;
    alive = bsz;
    while (alive) {
        if (tid < alive) {
            int li = step * (2 * tid + 1) - 1;
            int ri = step * (2 * tid + 2) - 1;
            sh_a[ri] += sh_a[li];
        }
        alive >>= 1;
        step <<= 1;
        __syncthreads();
    }

    if (tid == 0) {
        /* Save sum to global memory */
        gl_sums[bid] = sh_a[pref_sz - 1];
        sh_a[pref_sz - 1] = 0;
    }
    __syncthreads();

    /* leaves ------> root */
    alive = 1;
    step = bsz;
    while (alive <= bsz) {
        if (tid < alive) {
            int li = step * (2 * tid + 1) - 1;
            int ri = step * (2 * tid + 2) - 1;
            T tmp = sh_a[li];
            sh_a[li] = sh_a[ri];
            sh_a[ri] += tmp;
        }
        alive <<= 1;
        step >>= 1;
        __syncthreads();
    }

    /* Load data into global memory */
    gl_a[offset + 2 * tid] = sh_a[2 * tid];
    gl_a[offset + 2 * tid + 1] = sh_a[2 * tid + 1];
    __syncthreads();
}

template <class T>
__global__ void scan_kernel(T* gl_a, int gl_n, T* gl_sums) {
    extern __shared__ char sh_mem[];
    T* sh_a = (T*)sh_mem;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsz = blockDim.x;
    int pref_sz = 2 * bsz;
    int offset = pref_sz * bid;
    while (offset + 2 * tid + 1 < gl_n) {
        scan_block(gl_a, gl_sums, sh_a, tid, bid, bsz, pref_sz, offset);
        offset += SCAN_BLOCKS * pref_sz;
        bid += SCAN_BLOCKS;
    }
}

template <class T>
__global__ void update_kernel(T* gl_a, int gl_n, const T* gl_add) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsz = blockDim.x;
    int pref_sz = 2 * bsz;
    int offset = pref_sz * bid;
    while (offset + 2 * tid + 1 < gl_n) {
        gl_a[offset + 2 * tid] += gl_add[bid];
        gl_a[offset + 2 * tid + 1] += gl_add[bid];
        offset += SCAN_BLOCKS * pref_sz;
        bid += SCAN_BLOCKS;
        __syncthreads();
    }
}

template <class T>
void scan(const T* dev_a_in, T* dev_a_out, int n) {
    int lg_n = lg_2(n);
    int m = std::max(2, 1 << lg_n);

    /*
     * Make copy of size equals to the power of 2
     * and pad it with zeroes
     */
    T* dev_a_pad;
    CSC(cudaMalloc(&dev_a_pad, sizeof(T) * m));
    CSC(cudaMemcpy(dev_a_pad, dev_a_in, sizeof(T) * n,
                   cudaMemcpyDeviceToDevice));
    CSC(cudaMemset(dev_a_pad + n, 0, sizeof(T) * (m - n)));
    CSC(cudaGetLastError());

    /* dev_a_pad -----|scan|-----> dev_a_pad, dev_sums */
    T* dev_sums;
    int n_sums = std::max(1, m / (2 * SCAN_THREADS));
    CSC(cudaMalloc(&dev_sums, sizeof(T) * n_sums));

    int sh_mem = 2 * sizeof(T) * SCAN_THREADS;
    scan_kernel<<<SCAN_BLOCKS, SCAN_THREADS, sh_mem>>>(dev_a_pad, m, dev_sums);
    CSC(cudaGetLastError());

    /*
     * In case of big array
     * we need to call scan for sums array
     * and then update block partial sums
     */
    if (n_sums > 1) {
        T* dev_pref_sums;
        CSC(cudaMalloc(&dev_pref_sums, sizeof(T) * n_sums));
        scan(dev_sums, dev_pref_sums, n_sums);
        update_kernel<<<SCAN_BLOCKS, SCAN_THREADS>>>(dev_a_pad, m,
                                                     dev_pref_sums);
        CSC(cudaFree(dev_pref_sums));
        CSC(cudaGetLastError());
    }
    CSC(cudaMemcpy(dev_a_out, dev_a_pad, sizeof(T) * n,
                   cudaMemcpyDeviceToDevice));
    CSC(cudaFree(dev_a_pad));
    CSC(cudaFree(dev_sums));
    CSC(cudaGetLastError());
}

#endif /* SCAN */

#ifndef HIST
#define HIST

const int GL_HIST_BLOCKS = GLOBAL_BLOCKS;
const int GL_HIST_THREADS = GLOBAL_THREADS;

/*
 * If you need uint32_t or int64_t,
 * change this types
 */

using cnt_type_t = int;
using cnt_ptr = cnt_type_t*;

template <class T>
__global__ void cnt_kernel_gl(const T* a, int n, cnt_ptr gl_cnt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for (; idx < n; idx += offset) {
        atomicAdd(gl_cnt + a[idx], 1);
    }
}

template <class T>
void build_hist_global(const T* a, int n, cnt_ptr& cnt, int& cnt_sz) {
    cnt_kernel_gl<<<GL_HIST_BLOCKS, GL_HIST_THREADS>>>(a, n, cnt);
}

template <class T>
void build_hist(const T* a, int n, cnt_ptr& cnt, int& cnt_sz) {
    cnt_sz = max_val<T>(a, n) + 1;
    CSC(cudaMalloc(&cnt, sizeof(cnt_type_t) * cnt_sz));
    CSC(cudaMemset(cnt, 0, sizeof(cnt_type_t) * cnt_sz));
    build_hist_global(a, n, cnt, cnt_sz);
    CSC(cudaGetLastError());
}

#endif /* HIST */

#ifndef SORTS
#define SORTS

const int SORT_BLOCKS = GLOBAL_BLOCKS;
const int SORT_THREADS = GLOBAL_THREADS;
const int SORT_POCKET_SIZE = 2 * SORT_THREADS;
const int SORT_SPLIT_SIZE = 16;

/* This MAGIC trick moves max element from last bucket */
const float SMALL = 1e-6;
const float MAGIC = 1.0 - SMALL;

struct pocket {
    cnt_type_t pos;
    cnt_type_t len;

    pocket(cnt_type_t _pos, cnt_type_t _len) {
        pos = _pos;
        len = _len;
    }

    ~pocket() = default;
};

const cnt_type_t INF = 1e9;
using vec = std::vector<cnt_type_t>;
using vecp = std::vector<pocket>;

/*
 * It's more efficient to calculate
 * key each time, than load it from
 * global memory
 */
// #define key_id(elem, mn, mx, n_split) ((((elem) - (mn)) / ((mx) - (mn)) *
// MAGIC) * n_split)

__device__ int key_id(float elem, float mn, float mx, int n_split) {
    int res = ((elem - mn) / (mx - mn) * MAGIC * n_split);
    return res;
}

template <class T>
__global__ void key_kernel(const T* dev_in, int n, int* key, T min_el, T max_el,
                           int n_split) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for (; idx < n; idx += offset) {
        key[idx] = key_id(dev_in[idx], min_el, max_el, n_split);
    }
}

template <class T>
__global__ void distribution_kernel(const T* dev_in, T* dev_out, int n,
                                    cnt_type_t* cnt, T min_el, T max_el,
                                    int n_split) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for (; idx < n; idx += offset) {
        int key = key_id(dev_in[idx], min_el, max_el, n_split);
        dev_out[atomicAdd(cnt + key, 1)] = dev_in[idx];
    }
}

void process_pockets(const vec& pref, vecp& small, vecp& large) {
    cnt_type_t last_pos = 0;
    cnt_type_t last_len = pref[0];
    for (size_t i = 1; i < pref.size(); ++i) {
        cnt_type_t elem = pref[i] - pref[i - 1];
        if (last_len > SORT_POCKET_SIZE) {
            large.push_back(pocket(last_pos, last_len));
            last_pos += last_len;
            last_len = elem;
        } else if (last_len + elem <= SORT_POCKET_SIZE) {
            last_len += elem;
        } else {
            small.push_back(pocket(last_pos, last_len));
            last_pos += last_len;
            last_len = elem;
        }
    }
}

template <class T>
__global__ void odd_even_kernel(T* dev_in, T* dev_out, pocket* data,
                                int data_n) {
    extern __shared__ char sh_mem[];
    T* sh_a = (T*)sh_mem;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id1 = 2 * tid, id2 = 2 * tid + 1;

    while (bid < data_n) {
        int n = data[bid].len;
        int offset = data[bid].pos;
        /* Load data into shared memory */
        if (id1 < n) {
            sh_a[id1] = dev_in[offset + id1];
        }
        if (id2 < n) {
            sh_a[id2] = dev_in[offset + id2];
        }
        __syncthreads();

        /* odd or even phase? */
        int flag = 0;
        for (int i = 0; i < n; ++i) {
            if (id2 + flag < n) {
                if (sh_a[id1 + flag] > sh_a[id2 + flag]) {
                    cuda_swap(T, sh_a[id1 + flag], sh_a[id2 + flag]);
                }
            }
            flag ^= 1;
            __syncthreads();
        }

        /* Load data into global memory */
        if (2 * tid < n) {
            dev_out[offset + 2 * tid] = sh_a[2 * tid];
        }
        if (2 * tid + 1 < n) {
            dev_out[offset + 2 * tid + 1] = sh_a[2 * tid + 1];
        }

        __syncthreads();
        bid += SORT_BLOCKS;
    }
}

template <class T>
void bucket_sort(T*, T*, int);

template <class T>
void sort_large(T* dev_in, T* dev_out, const vecp& pockets) {
    for (pocket elem : pockets) {
        cnt_type_t p = elem.pos;
        cnt_type_t n = elem.len;
        bucket_sort(dev_in + p, dev_out + p, n);
        CSC(cudaGetLastError());
    }
}

template <class T>
void sort_small(T* dev_in, T* dev_out, const vecp& pockets) {
    int n = pockets.size();
    pocket* dev_p;
    CSC(cudaMalloc(&dev_p, sizeof(pockets) * n));
    CSC(cudaMemcpy(dev_p, pockets.data(), sizeof(pocket) * n,
                   cudaMemcpyHostToDevice));
    int sh_mem = sizeof(T) * SORT_POCKET_SIZE;
    odd_even_kernel<<<SORT_BLOCKS, SORT_THREADS, sh_mem>>>(dev_in, dev_out,
                                                           dev_p, n);
    CSC(cudaFree(dev_p));
    CSC(cudaGetLastError());
}

template <class T>
void bucket_sort(T* dev_in, T* dev_out, int n) {
    if (n == 0) {
        return;
    }
    T min_el = min_val<T>(dev_in, n);
    T max_el = max_val<T>(dev_in, n);
    if (same(min_el, max_el)) {
        CSC(cudaMemcpy(dev_out, dev_in, sizeof(T) * n,
                       cudaMemcpyDeviceToDevice));
        CSC(cudaGetLastError());
        return;
    }

    /* Calculate pocket keys */
    int n_split = (n + SORT_SPLIT_SIZE - 1) / SORT_SPLIT_SIZE;
    int* dev_key;
    CSC(cudaMalloc(&dev_key, sizeof(int) * n));
    key_kernel<<<SORT_BLOCKS, SORT_THREADS>>>(dev_in, n, dev_key, min_el,
                                              max_el, n_split);
    CSC(cudaGetLastError());

    /* Build histogram */
    cnt_type_t* dev_hist;
    int hist_sz = n_split;
    build_hist<int>(dev_key, n, dev_hist, hist_sz);
    CSC(cudaFree(dev_key));
    CSC(cudaGetLastError());

    /* Scan histogram */
    cnt_type_t* dev_pref;
    CSC(cudaMalloc(&dev_pref, sizeof(cnt_type_t) * hist_sz));
    scan<cnt_type_t>(dev_hist, dev_pref, hist_sz);
    CSC(cudaFree(dev_hist));
    CSC(cudaGetLastError());

    /* Distribute elements */
    distribution_kernel<<<SORT_BLOCKS, SORT_THREADS>>>(
        dev_in, dev_out, n, dev_pref, min_el, max_el, n_split);
    CSC(cudaMemcpy(dev_in, dev_out, sizeof(T) * n, cudaMemcpyDeviceToDevice));
    vec pref(hist_sz);
    CSC(cudaMemcpy(pref.data(), dev_pref, sizeof(cnt_type_t) * hist_sz,
                   cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_pref));
    CSC(cudaGetLastError());

    /*
     * Merge small pockets
     * Recursively sort large pockets
     * Sort small pockets using
     * shared memory odd-even sort
     */
    vecp small, large;
    pref.push_back(INF);
    process_pockets(pref, small, large);
    sort_large(dev_in, dev_out, large);
    sort_small(dev_in, dev_out, small);
    CSC(cudaGetLastError());
}

#endif /* SORTS */

#endif /* ALGO_HPP */
