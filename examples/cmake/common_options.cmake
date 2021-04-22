option(WITH_TIDY "turn on clang-tidy" OFF)

if (WITH_TIDY)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy;-header-filter=.*;)
endif()

#8 for double; 4 for float
set(REAL_SIZE_IN_BYTES "8" CACHE STRING "size of the floating point data type")
set_property(CACHE REAL_SIZE_IN_BYTES PROPERTY STRINGS "8" "4")

set(DEVICE_BACKEND "CUDA" CACHE STRING "type of an interface")
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS "CUDA" "HIP" "ONEAPI" "HIPSYCL")

set(DEVICE_SUB_ARCH "sm_60" CACHE STRING "model of Streaming Multiprocessor")
set(DEVICE_SUB_ARCH_OPTIONS "sm_20" "sm_30" "sm_50" "sm_60" "sm_70" "gfx906" "bdw" 
                            "skl" "kbl" "cfl" "bxt" "glk" "icllp" "lkf" "ehl" "tgllp" 
                            "rkl" "adls" "dg1" "Gen8" "Gen9" "Gen11" "Gen12LP")
set_property(CACHE DEVICE_SUB_ARCH PROPERTY STRINGS ${DEVICE_SUB_ARCH_OPTIONS})