#include "utils/logger.h"
#include "Internals.h"
#include <sstream>

namespace device {
  namespace internals {

    std::string PrevFile = "";
    int PrevLine = 0;

    void checkErr(const std::string &file, int line) {
      hipError_t Error = hipGetLastError();
      if (Error != hipSuccess) {
        std::stringstream stream;
        stream << '\n' << file << ", line " << line
               << ": " << hipGetErrorString(Error) << " (" << Error << ")\n";
        if (PrevLine > 0) {
          stream << "Previous HIP call:" << std::endl
                 << PrevFile << ", line " << PrevLine << std::endl;
        }
        logError() << stream.str();
      }
      PrevFile = file;
      PrevLine = line;
    }
  } // namespace internals
} // namespace device
