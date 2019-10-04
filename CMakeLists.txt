cmake_minimum_required (VERSION 2.6)
project (utests-yateto-gpu)

set(CMAKE_CXX_STANDARD 11)


# handle the user's input
set(PRECISION 8 CACHE STRING "num of bytes for floating point representation")
set(PRECISION_OPTIONS 4 8)
set_property(CACHE PRECISION PROPERTY STRINGS ${PRECISION_OPTIONS})

set(ALIGNMENT 64 CACHE STRING "num of bytes for floating point representation")
set(ALIGNMENT_OPTIONS 32 64)
set_property(CACHE ALIGNMENT PROPERTY STRINGS ${ALIGNMENT_OPTIONS})

message(STATUS "PRECISION: ${PRECISION}")
message(STATUS "ALIGNMENT: ${ALIGNMENT}")

set(HARDWARE_DEFINITIONS "ALIGNMENT={ALIGNMENT}"
                         "REAL_SIZE=${PRECISION}")

option(FAST_COMPARISON "compare floats without checking bits" ON)
if (${FAST_COMPARISON} STREQUAL ON)
    set(UNITE_TESTS_DEFINITIONS "FAST_COMPARISON")
    message(STATUS "FAST_COMPARISON: ON")
else()
    message(STATUS "FAST_COMPARISON: OFF")
endif()


# generate custom cuda gemm library
include_directories("cuda/src")
add_subdirectory(cuda) # compiles 'custom_blas' library


#===============================================================================
#                              UNIT TESTS
#===============================================================================
# collect unit tests source files
file(GLOB utest_host_src "tests/unit_tests/main.cpp"
                         "tests/aux/helper.cpp"
                         "tests/unit_tests/copy_add_scale/*.cpp"
                         "tests/unit_tests/single_gemms/*.cpp" 
                         "tests/unit_tests/multiple_gemms/*.cpp")
                  
# add and compile host side of tests
set(utest_target unit_tests)
add_executable(${utest_target} ${utest_host_src})
target_compile_options(${utest_target} PRIVATE "-g")
target_compile_definitions(${utest_target} PRIVATE ${HARDWARE_DEFINITIONS}
                                                   ${UNITE_TESTS_DEFINITIONS})
target_include_directories(${utest_target} PRIVATE "tests/aux")


# link all together
target_link_libraries(${utest_target} custom_blas
                                      -L${MKLROOT}/lib/intel64 
                                      -lmkl_intel_ilp64 
                                      -lmkl_intel_thread 
                                      -lmkl_core 
                                      -liomp5 
                                      -lpthread 
                                      -lm 
                                      -ldl)                                      

#===============================================================================
#                             PERFORMANCE TESTS
#===============================================================================
# collect performance tests source files
file(GLOB perf_host_src "tests/performance/*.cpp"
                        "tests/aux/helper.cpp")

set(perf_target performance_tests)
add_executable(${perf_target} ${perf_host_src})
target_compile_options(${perf_target} PRIVATE "-O2" "-fopenmp")
target_include_directories(${perf_target} PRIVATE "tests/aux")
target_compile_definitions(${perf_target} PRIVATE ${HARDWARE_DEFINITIONS})

# link all together
target_link_libraries(${perf_target} custom_blas
                                     -L${MKLROOT}/lib/intel64 
                                     -lmkl_intel_ilp64 
                                     -lmkl_intel_thread 
                                     -lmkl_core 
                                     -liomp5 
                                     -lpthread 
                                     -lm 
                                     -ldl
                                     -fopenmp)  