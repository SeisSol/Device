find_package(CUDA REQUIRED)
find_package(NVToolsExt)
if (NVToolsExt_FOUND)
    set(EXTRA_DEVICE_FLAGS -DPROFILING_ENABLED)
endif()

set(CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(device.cpp
                            algorithms/cuda/Reduction.cpp
                            PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT
                            OBJ)

# NOTE: cmake doesn't fully support $<$<COMPILE_LANGUAGE>:CUDA>:
set(CUDA_NVCC_FLAGS -std=c++14;
        -arch=${DEVICE_SUB_ARCH};
        -Xptxas -v;
        -DDEVICE_${DEVICE_BACKEND}_LANG;
        -DREAL_SIZE=${REAL_SIZE_IN_BYTES};
        --expt-relaxed-constexpr;
        ${EXTRA_DEVICE_FLAGS})

cuda_add_library(device device.cpp
                        interfaces/cuda/Control.cu
                        interfaces/cuda/Aux.cu
                        interfaces/cuda/Memory.cu
                        interfaces/cuda/Copy.cu
                        interfaces/cuda/Streams.cu
                        interfaces/cuda/Internals.cu
                        algorithms/cuda/ArrayManip.cu
                        algorithms/cuda/BatchManip.cu
                        algorithms/cuda/Debugging.cu
                        algorithms/cuda/Reduction.cpp)


target_link_directories(device PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
target_link_libraries(device -lcuda ${CUDA_LIBRARIES} ${NVToolsExt_LIBRARIES})
target_include_directories(device PUBLIC ${CUDA_INCLUDE_DIRS}
        ${NVToolsExt_INCLUDE_DIRS}
        ../../src)

target_compile_options(device PRIVATE "-std=c++11")
