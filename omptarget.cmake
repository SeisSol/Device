set(DEVICE_SOURCE_FILES device.cpp
algorithms/omptarget/ArrayManip.cpp
                        algorithms/omptarget/BatchManip.cpp
                        algorithms/omptarget/Reduction.cpp
        )

add_library(device STATIC ${DEVICE_SOURCE_FILES})

find_package(OmpTargetFlags REQUIRED)
target_link_libraries(device PRIVATE omptarget::device_flags)

# nvhpc: target_compile_options(device PRIVATE -mp=gpu -gpu=cc86,managed,zeroinit,ptxinfo,all,lineinfo -Minfo=mp)
