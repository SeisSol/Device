#Add the kernel as a lib
cmake_minimum_required(VERSION 3.5)
project(subroutinesKernel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME kernel)

if (${DEVICE_BACKEND} STREQUAL "HIP")

    #Compile the HIP kernel
    #Set up HIP
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()

    #set the CMAKE_MODULE_PATH for the helper cmake files from HIP
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

    set(HIP_COMPILER hcc)

    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()

    set(HIP_COMPILER hcc)

    #Set the source files and needed flags

    set(SOURCE_FILES subroutinesHIP.cpp)

    set(HIPCC_FLAGS -DREAL_SIZE=${REAL_SIZE_IN_BYTES})
    set(HCC_FLAGS)
    set(NVCC_FLAGS)

    set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_PATH} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

    set_source_files_properties(${SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

    #Compile and set the Linker language
    hip_add_library(${TARGET_NAME} SHARED ${SOURCE_FILES} HIPCC_OPTIONS ${HIPCC_FLAGS} HCC_OPTIONS ${HCC_FLAGS} NVCC_OPTIONS ${NVCC_FLAGS})


elseif(${DEVICE_BACKEND} STREQUAL "CUDA")

    #Compile the CUDA kernel
    find_package(CUDA REQUIRED)

    #Set the source files and needed flags
    set(SOURCE_FILES subroutinesCUDA.cu)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                        -DREAL_SIZE=${REAL_SIZE_IN_BYTES};
                        --compiler-options -fPIC;)

    #Compile
    cuda_add_library(${TARGET_NAME} ${SOURCE_FILES})

endif()
