cmake_minimum_required(VERSION 3.5)
project(device_submodule)

# check INTERFACE
if (NOT DEFINED DEVICE_BACKEND)
    message(FATAL_ERROR "DEVICE_BACKEND variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT CUDA DUMMY)
        if (${DEVICE_BACKEND} STREQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "DEVICE_BACKEND must be either CUDA or DUMMY. Given: ${DEVICE_BACKEND}")
    endif()
endif()


#check REAL_SIZE
if (NOT DEFINED REAL_SIZE_IN_BYTES)
    message(FATAL_ERROR "REAL_SIZE_IN_BYTES variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT 4 8)
        if (${REAL_SIZE_IN_BYTES} EQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "REAL_SIZE_IN_BYTES must be either 4 or 8. Given: ${REAL_SIZE}")
    endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
string(TOLOWER ${DEVICE_BACKEND} BACKEND_FODER)

if (${DEVICE_BACKEND} STREQUAL "DUMMY")
    add_library(device STATIC device.cpp
                              interfaces/dummy/DummyInterface.cpp)

elseif(${DEVICE_BACKEND} STREQUAL "CUDA")
    find_package(CUDA REQUIRED)
    find_package(NVToolsExt)
    if (NVToolsExt_FOUND)
        set(EXTRA_FLAGS -DPROFILING_ENABLED)
    endif()
    set(CUDA_SEPARABLE_COMPILATION ON)

    set_source_files_properties(device.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                        -std=c++11;
                        -arch=sm_50;
                        -Xptxas -v;
                        -D${DEVICE_BACKEND};
                        -DREAL_SIZE=${REAL_SIZE_IN_BYTES}
                        ${EXTRA_FLAGS})

    cuda_add_library(device STATIC device.cpp
                                   interfaces/cuda/Control.cu
                                   interfaces/cuda/Aux.cu
                                   interfaces/cuda/Memory.cu
                                   interfaces/cuda/Copy.cu
                                   interfaces/cuda/Streams.cu
                                   interfaces/cuda/Internals.cu
                                   interfaces/cuda/ComputeAddressing.cu
                                   interfaces/cuda/Gemm.cu
                                   interfaces/cuda/CopyAddScale.cu)

    target_link_libraries(device -lcuda ${CUDA_LIBRARIES} ${NVToolsExt_LIBRARIES})
    target_include_directories( device PRIVATE ${CUDA_INCLUDE_DIRS} ${NVToolsExt_INCLUDE_DIRS})

endif()

target_compile_definitions(device PUBLIC ${DEVICE_BACKEND} REAL_SIZE=${REAL_SIZE_IN_BYTES})
target_include_directories(device PUBLIC . interfaces/${BACKEND_FODER})
target_compile_options(device PUBLIC "-std=c++11")

