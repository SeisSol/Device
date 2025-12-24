// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/logger.h"
#include <sstream>
#include <string>
#include <unordered_set>

#include <cuda.h>

namespace device::internals {

thread_local std::string prevFile{};
thread_local int prevLine{-1};

cudaError_t checkResult(cudaError_t error, const std::string &file, int line, const std::unordered_set<cudaError_t>& except) {
  if (error != cudaSuccess && except.find(error) == except.end()) {
    std::stringstream stream;
    stream << '\n'
           << file << ", line " << line << ": " << cudaGetErrorString(error) << " (" << error
           << ")\n";
    if (prevLine >= 0) {
      stream << "Previous CUDA API/Driver call:" << std::endl
             << prevFile << ", line " << prevLine << std::endl;
    }
    logError() << stream.str();
  }
  prevFile = file;
  prevLine = line;
  return error;
}

CUresult checkResultDriver(CUresult error, const std::string& file, int line, const std::unordered_set<CUresult>& except) {
  if (error != CUDA_SUCCESS && except.find(error) == except.end()) {
    const char* errstr = nullptr;
    const auto errstrRes = cuGetErrorString(error, &errstr);

    std::stringstream stream;
    stream << '\n'
           << file << ", line " << line << ": ";
    if (errstrRes == CUDA_SUCCESS) {
      stream << errstr;
    }
    else {
      stream << "[ERROR WHILE RETRIEVING ERROR STRING]";
    }
    stream << " (" << error
           << ")\n";
    if (prevLine >= 0) {
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

