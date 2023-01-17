#ifndef IO_CUH
#define IO_CUH

#include <assert.h>
#include <stdio.h>

/* CheckSecureCode for fread system call */
#define csc_fread(ptr, sz, cnt, stream)                               \
    {                                                                 \
        size_t ret = fread((ptr), (sz), (cnt), (stream));             \
        if (ret != (size_t)(cnt)) {                                   \
            fprintf(stderr, "ERROR by %s:%d! Message: fread error\n", \
                    __FILE__, __LINE__);                              \
            exit(0);                                                  \
        }                                                             \
    }

uchar4* read_input(const char* fname, int& w, int& h) {
    FILE* file_stream = NULL;
    file_stream = fopen(fname, "rb");
    assert(file_stream != NULL);
    csc_fread(&w, sizeof(int), 1, file_stream);
    csc_fread(&h, sizeof(int), 1, file_stream);
    uchar4* ptr = new uchar4[w * h];
    csc_fread(ptr, sizeof(uchar4), w * h, file_stream);
    fclose(file_stream);
    return ptr;
}

void write_output(const char* fname, int w, int h, uchar4* ptr) {
    FILE* file_stream = NULL;
    file_stream = fopen(fname, "wb");
    assert(file_stream != NULL);
    fwrite(&w, sizeof(int), 1, file_stream);
    fwrite(&h, sizeof(int), 1, file_stream);
    fwrite(ptr, sizeof(uchar4), w * h, file_stream);
    fclose(file_stream);
}

#endif /* IO_CUH */
