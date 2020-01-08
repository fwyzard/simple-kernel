#include <cstdio>

#include <alpaka/alpaka.hpp>

#include "kernel.h"

template <typename T_Acc> ALPAKA_FN_ACC
void kernel::operator()(T_Acc const& acc, const char* message) const {
  printf("%s\n", message);
}

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template ALPAKA_FN_ACC
void kernel::operator()(CudaAcc_1d const& acc, const char* message) const;
#endif
