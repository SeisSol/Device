// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"

#include "utils/logger.h"

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <vector>

namespace device::internals {

thread_local const char* prevFile{nullptr};
thread_local int prevLine{0};

hipError_t checkResult(hipError_t error,
                       const char* file,
                       int line,
                       std::initializer_list<hipError_t> except) {
  if (error != hipSuccess && std::find(except.begin(), except.end(), error) == except.end()) {
    std::stringstream stream;
    stream << '\n'
           << file << ", line " << line << ": " << hipGetErrorString(error) << " (" << error
           << ")\n";
    if (prevFile != nullptr) {
      stream << "Previous HIP call:" << std::endl << prevFile << ", line " << prevLine << std::endl;
    }
    logError() << stream.str();
  }
  prevFile = file;
  prevLine = line;

  return error;
}

} // namespace device::internals
