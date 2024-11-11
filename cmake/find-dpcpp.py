# SPDX-FileCopyrightText: 2022-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from shutil import which


dpcpp = which("dpcpp")
clang = which("clang")
exe_name = dpcpp if dpcpp else clang

if exe_name:
  print(os.path.dirname(os.path.dirname(exe_name)), end = '')
else:
  sys.exit(1)

