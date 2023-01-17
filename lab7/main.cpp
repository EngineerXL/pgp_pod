#include <assert.h>
#include <math.h>
#include <mpi.h>

#include <iostream>

using namespace std;

#define to_ind(i, j, k) \
    (((i) + 1) * ((ny) + 2) + ((j) + 1) + ((k) + 1) * (((nx) + 2) * ((ny) + 2)))
#define to_indb(ib, jb, kb) ((ib) * (nby) + (jb) + (kb) * ((nbx) * (nby)))

const int PRECISION = 10;

/*
 * MPI_Sendrecv in 2 phases
 *
 * n = 5
 * phase = 0
 * | 0-1 0-1 0-|
 *
 * phase = 1
 * |-1 0-1 0-1 |
 *
 *
 * n = 4
 * phase = 0
 * | 0-1 0-1 |
 *
 * phase = 1
 * |-1 0-1 0-|
 */

#define exchange_up_down(ib, jb, kb, phase)                                    \
    {                                                                          \
        int exch_sz = (nx) * (ny);                                             \
        if (((kb) ^ (phase)) & 1) {                                            \
            if ((kb) > 0) {                                                    \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int j = 0; j < (ny); ++j) {                           \
                        buff_in[i * (ny) + j] = u_prev[to_ind(i, j, 0)];       \
                    }                                                          \
                }                                                              \
                MPI_Sendrecv(buff_in, exch_sz, MPI_DOUBLE,                     \
                             to_indb((ib), (jb), (kb)-1), id, buff_out,        \
                             exch_sz, MPI_DOUBLE, to_indb((ib), (jb), (kb)-1), \
                             to_indb((ib), (jb), (kb)-1), MPI_COMM_WORLD,      \
                             &status);                                         \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int j = 0; j < (ny); ++j) {                           \
                        u_prev[to_ind(i, j, -1)] = buff_out[i * (ny) + j];     \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int j = 0; j < (ny); ++j) {                           \
                        u_prev[to_ind(i, j, -1)] = u_down;                     \
                    }                                                          \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            if ((kb) + 1 < (nbz)) {                                            \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int j = 0; j < (ny); ++j) {                           \
                        buff_in[i * (ny) + j] = u_prev[to_ind(i, j, (nz)-1)];  \
                    }                                                          \
                }                                                              \
                MPI_Sendrecv(                                                  \
                    buff_in, exch_sz, MPI_DOUBLE,                              \
                    to_indb((ib), (jb), (kb) + 1), id, buff_out, exch_sz,      \
                    MPI_DOUBLE, to_indb((ib), (jb), (kb) + 1),                 \
                    to_indb((ib), (jb), (kb) + 1), MPI_COMM_WORLD, &status);   \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int j = 0; j < (ny); ++j) {                           \
                        u_prev[to_ind(i, j, (nz))] = buff_out[i * (ny) + j];   \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int j = 0; j < (ny); ++j) {                           \
                        u_prev[to_ind(i, j, (nz))] = u_up;                     \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

#define exchange_front_back(ib, jb, kb, phase)                                 \
    {                                                                          \
        int exch_sz = (nx) * (nz);                                             \
        if (((jb) ^ (phase)) & 1) {                                            \
            if ((jb) > 0) {                                                    \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        buff_in[i * (nz) + k] = u_prev[to_ind(i, 0, k)];       \
                    }                                                          \
                }                                                              \
                MPI_Sendrecv(buff_in, exch_sz, MPI_DOUBLE,                     \
                             to_indb((ib), (jb)-1, (kb)), id, buff_out,        \
                             exch_sz, MPI_DOUBLE, to_indb((ib), (jb)-1, (kb)), \
                             to_indb((ib), (jb)-1, (kb)), MPI_COMM_WORLD,      \
                             &status);                                         \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind(i, -1, k)] = buff_out[i * (nz) + k];     \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind(i, -1, k)] = u_front;                    \
                    }                                                          \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            if ((jb) + 1 < (nby)) {                                            \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        buff_in[i * (nz) + k] = u_prev[to_ind(i, (ny)-1, k)];  \
                    }                                                          \
                }                                                              \
                MPI_Sendrecv(                                                  \
                    buff_in, exch_sz, MPI_DOUBLE,                              \
                    to_indb((ib), (jb) + 1, (kb)), id, buff_out, exch_sz,      \
                    MPI_DOUBLE, to_indb((ib), (jb) + 1, (kb)),                 \
                    to_indb((ib), (jb) + 1, (kb)), MPI_COMM_WORLD, &status);   \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind(i, (ny), k)] = buff_out[i * (nz) + k];   \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int i = 0; i < (nx); ++i) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind(i, (ny), k)] = u_back;                   \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

#define exchange_left_right(ib, jb, kb, phase)                                 \
    {                                                                          \
        int exch_sz = (ny) * (nz);                                             \
        if (((ib) ^ (phase)) & 1) {                                            \
            if ((ib) > 0) {                                                    \
                for (int j = 0; j < (ny); ++j) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        buff_in[j * (nz) + k] = u_prev[to_ind(0, j, k)];       \
                    }                                                          \
                }                                                              \
                MPI_Sendrecv(buff_in, exch_sz, MPI_DOUBLE,                     \
                             to_indb((ib)-1, (jb), (kb)), id, buff_out,        \
                             exch_sz, MPI_DOUBLE, to_indb((ib)-1, (jb), (kb)), \
                             to_indb((ib)-1, (jb), (kb)), MPI_COMM_WORLD,      \
                             &status);                                         \
                for (int j = 0; j < (ny); ++j) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind(-1, j, k)] = buff_out[j * (nz) + k];     \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int j = 0; j < (ny); ++j) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind(-1, j, k)] = u_left;                     \
                    }                                                          \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            if ((ib) + 1 < (nbx)) {                                            \
                for (int j = 0; j < (ny); ++j) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        buff_in[j * (nz) + k] = u_prev[to_ind((nx)-1, j, k)];  \
                    }                                                          \
                }                                                              \
                MPI_Sendrecv(                                                  \
                    buff_in, exch_sz, MPI_DOUBLE,                              \
                    to_indb((ib) + 1, (jb), (kb)), id, buff_out, exch_sz,      \
                    MPI_DOUBLE, to_indb((ib) + 1, (jb), (kb)),                 \
                    to_indb((ib) + 1, (jb), (kb)), MPI_COMM_WORLD, &status);   \
                for (int j = 0; j < (ny); ++j) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind((nx), j, k)] = buff_out[j * (nz) + k];   \
                    }                                                          \
                }                                                              \
            } else {                                                           \
                for (int j = 0; j < (ny); ++j) {                               \
                    for (int k = 0; k < (nz); ++k) {                           \
                        u_prev[to_ind((nx), j, k)] = u_right;                  \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

#define print_ans()                                                            \
    {                                                                          \
        cout.precision(PRECISION);                                             \
        cout << scientific;                                                    \
        if (id == 0) {                                                         \
            for (int kb = 0; kb < nbz; ++kb) {                                 \
                for (int k = 0; k < nz; ++k) {                                 \
                    if (k + kb) {                                              \
                        cout << '\n';                                          \
                    }                                                          \
                    for (int jb = 0; jb < nby; ++jb) {                         \
                        for (int j = 0; j < ny; ++j) {                         \
                            for (int ib = 0; ib < nbx; ++ib) {                 \
                                if (ib) {                                      \
                                    cout << ' ';                               \
                                }                                              \
                                int idb = to_indb(ib, jb, kb);                 \
                                if (idb == 0) {                                \
                                    for (int i = 0; i < nx; ++i) {             \
                                        buff_out[i] = u_prev[to_ind(i, j, k)]; \
                                    }                                          \
                                } else {                                       \
                                    MPI_Recv(buff_out, nx, MPI_DOUBLE, idb,    \
                                             idb, MPI_COMM_WORLD, &status);    \
                                }                                              \
                                for (int i = 0; i < nx; ++i) {                 \
                                    if (i) {                                   \
                                        cout << ' ';                           \
                                    }                                          \
                                    cout << buff_out[i];                       \
                                }                                              \
                            }                                                  \
                            cout << '\n';                                      \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            for (int k = 0; k < nz; ++k) {                                     \
                for (int j = 0; j < ny; ++j) {                                 \
                    for (int i = 0; i < nx; ++i) {                             \
                        buff_in[i] = u_prev[to_ind(i, j, k)];                  \
                    }                                                          \
                    MPI_Send(buff_in, nx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);  \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

const double INF = 1e18;

#define CHECK_PGP
// #define BENCH

int main(int argc, char* argv[]) {
    int id, numproc, proc_name_len;
    char proc_name[MPI_MAX_PROCESSOR_NAME];

    int nx, ny, nz;
    int nbx, nby, nbz;
    string fname;
    double eps;
    double lx, ly, lz;
    double u_down, u_up, u_left, u_right, u_front, u_back, u0;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Get_processor_name(proc_name, &proc_name_len);

    if (id == 0) {
        cin >> nbx >> nby >> nbz;
        cin >> nx >> ny >> nz;
        cin >> fname;
#ifdef CHECK_PGP
        FILE* f = freopen(fname.c_str(), "w", stdout);
        if (f == NULL) {
            cerr << "Can not open file!" << endl;
            return 0;
        }
#endif
        cin >> eps;
        cin >> lx >> ly >> lz;
        cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back;
        cin >> u0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int z_layer_sz = nbx * nby;
    int idx = (id % z_layer_sz) / nby;
    int idy = (id % z_layer_sz) % nby;
    int idz = id / z_layer_sz;

    double hx = lx / (nx * nbx);
    double hy = ly / (ny * nby);
    double hz = lz / (nz * nbz);

    double hx_2 = 1 / (hx * hx);
    double hy_2 = 1 / (hy * hy);
    double hz_2 = 1 / (hz * hz);

    int max_dim = max(nx, max(ny, nz));
    int max_layer_sz = (max_dim + 2) * (max_dim + 2);
    double* u_prev = new double[(nx + 2) * (ny + 2) * (nz + 2)];
    double* u_next = new double[(nx + 2) * (ny + 2) * (nz + 2)];
    double* buff_in = new double[max_layer_sz];
    double* buff_out = new double[max_layer_sz];
    double* deltas = new double[nbx * nby * nbz];

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; ++k) {
                u_prev[to_ind(i, j, k)] = u0;
            }
        }
    }

#ifdef BENCH
    double start = MPI_Wtime();
#endif
    double delta = INF;
    while (delta > eps) {
        MPI_Barrier(MPI_COMM_WORLD);
        exchange_up_down(idx, idy, idz, 0);
        exchange_up_down(idx, idy, idz, 1);
        exchange_front_back(idx, idy, idz, 0);
        exchange_front_back(idx, idy, idz, 1);
        exchange_left_right(idx, idy, idz, 0);
        exchange_left_right(idx, idy, idz, 1);
        MPI_Barrier(MPI_COMM_WORLD);
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
                    delta = max(delta, fabs(u_next[to_ind(i, j, k)] -
                                            u_prev[to_ind(i, j, k)]));
                }
            }
        }
        MPI_Allgather(&delta, 1, MPI_DOUBLE, deltas, 1, MPI_DOUBLE,
                      MPI_COMM_WORLD);
        for (int id = 0; id < nbx * nby * nbz; ++id) {
            delta = max(delta, deltas[id]);
        }
        swap(u_prev, u_next);
    }
#ifdef BENCH
    double end = MPI_Wtime();
    if (id == 0) {
        cout.precision(3);
        cout << fixed << (end - start) * 1e3 << endl;
    }
#else
    print_ans();
#endif

    delete[] u_prev;
    delete[] u_next;
    delete[] buff_in;
    delete[] buff_out;
    delete[] deltas;
    MPI_Finalize();
    return 0;
}