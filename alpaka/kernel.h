#include <alpaka/alpaka.hpp>

#include "alpaka.h"

struct kernel {
  template <typename T_Acc> ALPAKA_FN_ACC
  void operator()(T_Acc const& acc, const char* message) const;
};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
extern template ALPAKA_FN_ACC
void kernel::operator()(CudaAcc_1d const& acc, const char* message) const;
#endif
