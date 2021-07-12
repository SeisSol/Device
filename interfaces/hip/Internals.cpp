#include "utils/logger.h"
#include "Internals.h"
#include <sstream>

namespace device {
  namespace internals {

    std::string prevFile{};
    int prevLine = 0;

    void checkErr(const std::string &file, int line) {
#ifndef NDEBUG
      hipError_t error = hipGetLastError();
      if (error != hipSuccess) {
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
#endif
    }
  } // namespace internals
} // namespace device
