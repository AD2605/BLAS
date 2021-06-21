#ifndef MATRIXMULTIPLICATION_DEFINES_CUH
#define MATRIXMULTIPLICATION_DEFINES_CUH

inline void gpuAssert(cudaError_t cudaError, const char* file. int line, bool abort = true){
    if(cudaError != cudaSuccess){
        fprintf(stderr, "Device Side Error : %s %s %d \n", cudaGetErrorString(cudaError), file, line);
        if(abort){
            exit(cudaError);
        }
    }
}

#define GPUCheckUtil(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif //MATRIXMULTIPLICATION_DEFINES_CUH
