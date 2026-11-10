// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Trace.h"

namespace device::hsaqueues {

DagLog& dagLog() {
  static DagLog log;
  return log;
}

Tracer& tracer() {
  static Tracer tracer;
  return tracer;
}

} // namespace device::hsaqueues
