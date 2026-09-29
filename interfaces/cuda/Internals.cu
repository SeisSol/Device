// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/logger.h"

#include <algorithm>
#include <cuda.h>
#include <initializer_list>
#include <sstream>
#include <string>

namespace device::internals {

thread_local const char* prevFile{nullptr};
thread_local int prevLine{-1};

cudaError_t checkResult(cudaError_t error,
                        const char* file,
                        int line,
                        std::initializer_list<cudaError_t> except) {
  if (error != cudaSuccess && std::find(except.begin(), except.end(), error) == except.end()) {
    std::stringstream stream;
    stream << '\n'
           << file << ", line " << line << ": " << cudaGetErrorString(error) << " (" << error
           << ")\n";
    if (prevFile != nullptr) {
      stream << "Previous CUDA API/Driver call:" << std::endl
             << prevFile << ", line " << prevLine << std::endl;
    }
    logError() << stream.str();
  }
  prevFile = file;
  prevLine = line;
  return error;
}

CUresult checkResultDriver(CUresult error,
                           const char* file,
                           int line,
                           std::initializer_list<CUresult> except) {
  if (error != CUDA_SUCCESS && std::find(except.begin(), except.end(), error) == except.end()) {
    const char* errstr = nullptr;
    const auto errstrRes = cuGetErrorString(error, &errstr);

    std::stringstream stream;
    stream << '\n' << file << ", line " << line << ": ";
    if (errstrRes == CUDA_SUCCESS) {
      stream << errstr;
    } else {
      stream << "[ERROR WHILE RETRIEVING ERROR STRING]";
    }
    stream << " (" << error << ")\n";
    if (prevFile != nullptr) {
      stream << "Previous CUDA API/Driver call:" << std::endl
             << prevFile << ", line " << prevLine << std::endl;
    }
    logError() << stream.str();
  }
  prevFile = file;
  prevLine = line;
  return error;
}

} // namespace device::internals
