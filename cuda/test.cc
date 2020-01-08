#include <string>

#include <cuda_runtime.h>

#include "cute/check.h"
#include "cute/launch.h"
#include "kernel.h"

int main(void) {
  const std::string msg = "Hello CUDA";

  char* msg_d;
  CUTE_CHECK(cudaMalloc(&msg_d, msg.size() + 1));
  CUTE_CHECK(cudaMemcpy(msg_d, msg.data(), msg.size() + 1, cudaMemcpyDefault));

  cute::launch(kernel, {1, 1}, msg_d);

  CUTE_CHECK(cudaDeviceSynchronize());

  return 0;
}
