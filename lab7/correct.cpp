#include <bits/stdc++.h>

using namespace std;

const double INF = 1e18;

#define to_ind(i, j, k) \
    (((i) + 1) * ((ny) + 2) + ((j) + 1) + ((k) + 1) * (((nx) + 2) * ((ny) + 2)))

int main() {
    int nx, ny, nz;
    int nbx, nby, nbz;
    string fname;
    double eps;
    double lx, ly, lz;
    double u_down, u_up, u_left, u_right, u_front, u_back, u0;

    cin >> nbx >> nby >> nbz;
    cin >> nx >> ny >> nz;
    cin >> fname;
    cin >> eps;
    cin >> lx >> ly >> lz;
    cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back;
    cin >> u0;

    double hx = lx / (nx * nbx);
    double hy = ly / (ny * nby);
    double hz = lz / (nz * nbz);

    double hx_2 = 1 / (hx * hx);
    double hy_2 = 1 / (hy * hy);
    double hz_2 = 1 / (hz * hz);
    nx *= nbx;
    ny *= nby;
    nz *= nbz;

    double* u_prev = new double[(nx + 2) * (ny + 2) * (nz + 2)];
    double* u_next = new double[(nx + 2) * (ny + 2) * (nz + 2)];
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                u_prev[to_ind(i, j, k)] = u0;
            }
        }
    }
    for (int i = 0; i < (nx); ++i) {
        for (int j = 0; j < (ny); ++j) {
            u_prev[to_ind(i, j, -1)] = u_down;
            u_next[to_ind(i, j, -1)] = u_down;
        }
    }
    for (int i = 0; i < (nx); ++i) {
        for (int j = 0; j < (ny); ++j) {
            u_prev[to_ind(i, j, (nz))] = u_up;
            u_next[to_ind(i, j, (nz))] = u_up;
        }
    }
    for (int i = 0; i < (nx); ++i) {
        for (int k = 0; k < (nz); ++k) {
            u_prev[to_ind(i, -1, k)] = u_front;
            u_next[to_ind(i, -1, k)] = u_front;
        }
    }
    for (int i = 0; i < (nx); ++i) {
        for (int k = 0; k < (nz); ++k) {
            u_prev[to_ind(i, (ny), k)] = u_back;
            u_next[to_ind(i, (ny), k)] = u_back;
        }
    }
    for (int j = 0; j < (ny); ++j) {
        for (int k = 0; k < (nz); ++k) {
            u_prev[to_ind(-1, j, k)] = u_left;
            u_next[to_ind(-1, j, k)] = u_left;
        }
    }
    for (int j = 0; j < (ny); ++j) {
        for (int k = 0; k < (nz); ++k) {
            u_prev[to_ind((nx), j, k)] = u_right;
            u_next[to_ind((nx), j, k)] = u_right;
        }
    }
    double delta = INF;
    int iters = 0;
    while (delta > eps) {
        ++iters;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    u_next[to_ind(i, j, k)] = ((u_prev[to_ind(i + 1, j, k)] +
                                                u_prev[to_ind(i - 1, j, k)]) *
                                                   hx_2 +
                                               (u_prev[to_ind(i, j + 1, k)] +
                                                u_prev[to_ind(i, j - 1, k)]) *
                                                   hy_2 +
                                               (u_prev[to_ind(i, j, k + 1)] +
                                                u_prev[to_ind(i, j, k - 1)]) *
                                                   hz_2) /
                                              (2 * (hx_2 + hy_2 + hz_2));
                }
            }
        }
        delta = -INF;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    delta = max(delta, abs(u_next[to_ind(i, j, k)] -
                                           u_prev[to_ind(i, j, k)]));
                }
            }
        }
        swap(u_next, u_prev);
    }
    cout.precision(10);
    cout << scientific;
    for (int k = 0; k < nz; ++k) {
        if (k) {
            cout << '\n';
        }
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (i) {
                    cout << ' ';
                }
                cout << u_prev[to_ind(i, j, k)];
            }
            cout << '\n';
        }
    }
    // cout << iters << endl;

    delete[] u_prev;
    delete[] u_next;
}
