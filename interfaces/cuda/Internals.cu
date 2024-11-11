// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/logger.h"
#include <sstream>

namespace device {
namespace internals {

std::string prevFile{};
int prevLine = 0;

void checkErr(const std::string &file, int line) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream stream;
    stream << '\n'
           << file << ", line " << line << ": " << cudaGetErrorString(error) << " (" << error
           << ")\n";
    if (prevLine > 0) {
      stream << "Previous CUDA call:" << std::endl
             << prevFile << ", line " << prevLine << std::endl;
    }
    logError() << stream.str();
  }
  prevFile = file;
  prevLine = line;
}
} // namespace internals
} // namespace device

