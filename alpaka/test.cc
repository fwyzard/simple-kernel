#include <string>

#include <alpaka/alpaka.hpp>

#include "alpaka.h"
#include "kernel.h"

int main(void) {
  const Host          host   = alpaka::pltf::getDevByIdx<HostPlatform>(0);
  const CudaDevice_1d device = alpaka::pltf::getDevByIdx<CudaPltform_1d>(0);

  CudaQueue queue(device);

  const std::string msg = "Hello CUDA";
  Extent size = msg.size() + 1;

  auto msg_hBuf = alpaka::mem::view::ViewPlainPtr<Host, const char, Dim_1d, Idx>(msg.data(), host, size);
  auto msg_dBuf = alpaka::mem::buf::alloc<char, Idx>(device, size);
  char* msg_d = alpaka::mem::view::getPtrNative(msg_dBuf);

  alpaka::mem::view::copy(queue, msg_dBuf, msg_hBuf, size);

  const WorkDiv_1d grid(
      Vec_1d::all(1),
      Vec_1d::all(1),
      Vec_1d::all(1));

  alpaka::queue::enqueue(queue, alpaka::kernel::createTaskKernel<CudaAcc_1d>(grid, kernel(), msg_d));

  return 0;
}
