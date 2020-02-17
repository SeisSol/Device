// NOTE: DO NOT INCLUDE THIS FILE INTO OTHER HEADERS.
// USE IT ONLY WITHIN AN IMPLEMENTATION

#ifndef SEISSOL_INTERNALS_H
#define SEISSOL_INTERNALS_H

#include <string>

#define CHECK_ERR device::internals::checkErr(__FILE__,__LINE__)
namespace device {
  namespace internals {
    void checkErr(const std::string &file, int line);
  }
}

#endif //SEISSOL_INTERNALS_H
