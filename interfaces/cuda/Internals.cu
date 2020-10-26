#include <iostream>
#include <cuda_runtime.h>
#include <sstream>

namespace device {
  namespace internals {

    std::string PrevFile = "";
    int PrevLine = 0;

    void checkErr(const std::string &file, int line) {
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {
        std::stringstream stream;
        stream << '\n' << file << ", line " << line
                  << ": " << cudaGetErrorString(Error) << " (" << Error << ")\n";
        if (PrevLine > 0) {
          stream << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        }
        throw std::runtime_error(stream.str());
      }
      PrevFile = file;
      PrevLine = line;
    }
  }
}

