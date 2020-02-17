#include <iostream>
#include <cuda_runtime.h>

namespace device {
  namespace internals {

    std::string PrevFile = "";
    int PrevLine = 0;

    void checkErr(const std::string &file, int line) {
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {
        std::cout << std::endl << file << ", line " << line
                  << ": " << cudaGetErrorString(Error) << " (" << Error << ")" << std::endl;
        if (PrevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }
      PrevFile = file;
      PrevLine = line;
    }
  }
}

