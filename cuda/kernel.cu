#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void kernel(const char* message) {
  printf("%s\n", message);
}
