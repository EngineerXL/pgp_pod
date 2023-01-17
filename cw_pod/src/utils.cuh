#ifndef UTILS_CUH
#define UTILS_CUH

/* GPU info */
void device_info() {
    int deviceCount;
    cudaDeviceProp devProp;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d devices\n", deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaGetDeviceProperties(&devProp, device);
        printf("Device %d\n", device);
        printf("Compute capability      : %d.%d\n", devProp.major,
               devProp.minor);
        printf("Name                    : %s\n", devProp.name);
        printf("Total Global Memory     : %zu\n", devProp.totalGlobalMem);
        printf("Shared memory per block : %d\n",
               (int)devProp.sharedMemPerBlock);
        printf("Registers per block     : %d\n", devProp.regsPerBlock);
        printf("Warp size               : %d\n", devProp.warpSize);
        printf("Max threads per block   : (%d, %d, %d)\n",
               devProp.maxThreadsDim[0], devProp.maxThreadsDim[1],
               devProp.maxThreadsDim[2]);
        printf("Max block               : (%d, %d, %d)\n",
               devProp.maxGridSize[0], devProp.maxGridSize[1],
               devProp.maxGridSize[2]);
        printf("Total constant memory   : %d\n", (int)devProp.totalConstMem);
        printf("Multiprocessors count   : %d\n", devProp.multiProcessorCount);
    }
}

/* CheckSecureCode */
#define CSC(call)                                                      \
    {                                                                  \
        cudaError_t status = (call);                                   \
        if (status != cudaSuccess) {                                   \
            const char* msg = cudaGetErrorString(status);              \
            fprintf(stderr, "ERROR by %s:%d! Message: %s\n", __FILE__, \
                    __LINE__, msg);                                    \
            exit(0);                                                   \
        }                                                              \
    }

struct cuda_timer_t {
    float ms;
    cudaEvent_t tstart, tstop;

    cuda_timer_t() {
        CSC(cudaEventCreate(&tstart));
        CSC(cudaEventCreate(&tstop));
    }

    void start() { CSC(cudaEventRecord(tstart)); }

    void end() {
        CSC(cudaDeviceSynchronize());
        CSC(cudaGetLastError());
        CSC(cudaEventRecord(tstop));
        CSC(cudaEventSynchronize(tstop));
    }

    float get_time() {
        CSC(cudaEventElapsedTime(&ms, tstart, tstop));
        return ms;
    }

    ~cuda_timer_t() {
        CSC(cudaEventDestroy(tstart));
        CSC(cudaEventDestroy(tstop));
    }
};

#endif /* UTILS_CUH */
