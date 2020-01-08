#ifndef cute_check_h
#define cute_check_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

namespace cute {

  [[noreturn]] inline void abortOnCudaError(const char* file,
                                            int line,
                                            const char* cmd,
                                            const char* error,
                                            const char* message,
                                            const char* description = nullptr) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "CUTE_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (description)
      out << description << "\n";
    throw std::runtime_error(out.str());
  }

  // check the result of a CUDA Driver function
  inline bool check_(
      const char* file, int line, const char* cmd, CUresult result, const char* description = nullptr) {
    if (__builtin_expect(result == CUDA_SUCCESS, true))
      return true;

    const char* error;
    const char* message;
    cuGetErrorName(result, &error);
    cuGetErrorString(result, &message);
    abortOnCudaError(file, line, cmd, error, message, description);
    return false;
  }

  // check the result of a CUDA Runtime function
  inline bool check_(
      const char* file, int line, const char* cmd, cudaError_t result, const char* description = nullptr) {
    if (__builtin_expect(result == cudaSuccess, true))
      return true;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    abortOnCudaError(file, line, cmd, error, message, description);
    return false;
  }

}  // namespace cute

#define CUTE_CHECK(ARG, ...) (cute::check_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // cute_check_h
