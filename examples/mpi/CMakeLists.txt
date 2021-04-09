cmake_minimum_required(VERSION 3.13)
project(MPI)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
find_package(IntelMPI REQUIRED)

add_executable(mpi src/main.cpp)
add_subdirectory(root)
target_include_directories(mpi PUBLIC root)
target_link_libraries(mpi PUBLIC device)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    target_link_options(mpi PRIVATE ${MPI_CXX_LINK_FLAGS})
else()
    set_target_properties(mpi PROPERTIES LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endif()
target_compile_options(mpi PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
target_compile_definitions(mpi PRIVATE DEVICE_${DEVICE_BACKEND}_LANG REAL_SIZE=${REAL_SIZE_IN_BYTES})
