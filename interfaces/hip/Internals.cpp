// SPDX-FileCopyrightText: 2020-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/logger.h"
#include "Internals.h"
#include <sstream>
#include <vector>
#include <unordered_set>

namespace device::internals {

thread_local std::string prevFile{};
thread_local int prevLine{0};

hipError_t checkResult(hipError_t error, const std::string &file, int line, const std::unordered_set<hipError_t>& except) {
  if (error != hipSuccess && except.find(error) == except.end()) {
    std::stringstream stream;
    stream << '\n' << file << ", line " << line
            << ": " << hipGetErrorString(error) << " (" << error << ")\n";
    if (prevLine > 0) {
      stream << "Previous HIP call:" << std::endl
              << prevFile << ", line " << prevLine << std::endl;
    }
    logError() << stream.str();
  }
  prevFile = file;
  prevLine = line;

  return error;
}

} // namespace device::internals

