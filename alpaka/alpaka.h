#ifndef alpaka_h
#define alpaka_h

#include <alpaka/alpaka.hpp>

// common types
using Idx = int32_t;
using Extent = int32_t;

// host platform
using Host = alpaka::dev::DevCpu;
using HostPlatform = alpaka::pltf::Pltf<Host>;

// 1-dimensional layout
using Dim_1d = alpaka::dim::DimInt<1u>;
using Vec_1d = alpaka::vec::Vec<Dim_1d, Idx>;
using WorkDiv_1d = alpaka::workdiv::WorkDivMembers<Dim_1d, Idx>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
// CUDA platform (1d)
using CudaAcc_1d = alpaka::acc::AccGpuCudaRt<Dim_1d, Extent>;
using CudaDevice_1d = alpaka::dev::Dev<CudaAcc_1d>;
using CudaPltform_1d = alpaka::pltf::Pltf<CudaDevice_1d>;

// use blocking operations
using CudaQueue = alpaka::queue::QueueCudaRtBlocking;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#endif  // alpaka_h
