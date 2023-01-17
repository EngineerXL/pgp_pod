#include <iostream>

#include "algo.cuh"

using namespace std;

// #define CHECK_PGP

using data_t = float;

#define BENCHMARK

int main() {
    // device_info();

    int n;
#ifdef CHECK_PGP
    csc_fread(&n, sizeof(int), 1, stdin);
#else
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> n;
#endif /* CHECK_PGP */

    data_t* a = new data_t[n];
#ifdef CHECK_PGP
    for (int i = 0; i < n; ++i) {
        csc_fread(&a[i], sizeof(data_t), 1, stdin);
    }
#else
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
#endif /* CHECK_PGP */

    data_t* dev_a;
    CSC(cudaMalloc(&dev_a, sizeof(data_t) * n));
    CSC(cudaMemcpy(dev_a, a, sizeof(data_t) * n, cudaMemcpyHostToDevice));

    data_t* dev_ans;
    CSC(cudaMalloc(&dev_ans, sizeof(data_t) * n));
    cuda_timer_t tmr;
    tmr.start();
    bucket_sort<data_t>(dev_a, dev_ans, n);
    tmr.end();

    CSC(cudaMemcpy(a, dev_ans, sizeof(data_t) * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_ans));
    CSC(cudaFree(dev_a));

#ifdef BENCHMARK
    cout.precision(3);
    cout << fixed;
    cout << tmr.get_time() << endl;
#endif /* CHECK_PGP */

#ifdef CHECK_PGP
    for (int i = 0; i < n; ++i) {
        fwrite(&a[i], sizeof(data_t), 1, stdout);
    }
#else
#ifndef BENCHMARK
    cout.precision(3);
    cout << fixed;
    for (int i = 0; i < n; ++i) {
        if (i) {
            cout << ' ';
        }
        cout << a[i];
    }
    cout << '\n';
#endif /* BENCHMARK */
#endif /* CHECK_PGP */

    delete [] a;
    return 0;
}
