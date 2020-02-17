cmake_minimum_required(VERSION 3.5)
project(device_submodule)

# check INTERFACE
if (NOT DEFINED INTERFACE)
    message(FATAL_ERROR "INTERFACE variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT CUDA DUMMY)
        if (${INTERFACE} STREQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "INTERFACE must be either CUDA or DUMMY. Given: ${INTERFACE}")
    endif()
endif()


#check REAL_SIZE
if (NOT DEFINED REAL_SIZE)
    message(FATAL_ERROR "REAL_SIZE variable has not been provided into the submodule")
else()
    set(FOUND OFF)
    foreach(VARIANT 4 8)
        if (${REAL_SIZE} EQUAL ${VARIANT})
            set(FOUND ON)
        endif()
    endforeach()
    if (NOT FOUND)
        message(FATAL_ERROR "REAL_SIZE must be either 4 or 8. Given: ${REAL_SIZE}")
    endif()
endif()

string(TOLOWER ${INTERFACE} INTERFACE_FODER)

if (${INTERFACE} STREQUAL "DUMMY")
    add_library(device STATIC device.cpp
                              interfaces/dummy/DummyInterface.cpp)

elseif(${INTERFACE} STREQUAL "CUDA")
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -arch=sm_50; -Xptxas -v; -D${INTERFACE}; -DREAL_SIZE=${REAL_SIZE})
    set_source_files_properties(device.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    cuda_add_library(device STATIC device.cpp
                                   interfaces/cuda/Control.cu
                                   interfaces/cuda/Aux.cu
                                   interfaces/cuda/Memory.cu
                                   interfaces/cuda/Copy.cu
                                   interfaces/cuda/Internals.cu)

    target_link_libraries(device -lcuda)
    target_include_directories(device PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

target_compile_definitions(device PUBLIC ${INTERFACE} REAL_SIZE=${REAL_SIZE})
target_include_directories(device PUBLIC . interfaces/${INTERFACE_FODER})
target_compile_options(device PUBLIC "-std=c++11")

