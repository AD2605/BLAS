This repository is a study of GPU architecture via implementing various BLAS subroutines 

All Level 1, Level 2 and Level 3 routines would be implemented in both Cuda and SYCL(and one day Vulkan)

Currently the following have been implemented - 

### Matrix Multiplication
* Naive Matrix Multiplication
* Shared, Coalesced with No Bank Conflict Matrix Addition
* Optimization in terms of number of instructions via outer product

### Level 1 BLAS
* Dot Product (A hybrid CPU GPU algorithm)
* Array/ Matrix Copy
* Matrix Transpose
