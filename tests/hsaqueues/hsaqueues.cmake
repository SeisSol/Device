# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

# The test of HSA queues and the probes that measure what an architecture takes (see README.md in
# this directory), with DEVICE_HSA_QUEUE_TESTS=ON; included by hip.cmake.

set(HSA_QUEUE_TEST_DIR ${CMAKE_CURRENT_LIST_DIR})
set(HSA_QUEUE_HIPCC -std=c++17 -O2 --offload-arch=${DEVICE_ARCH})

# the test: kernels, copies, events across queues, graphs, host functions, the mix with HIP streams,
# waits for the host; through the Device library and the launch hook
set_source_files_properties(${HSA_QUEUE_TEST_DIR}/HsaQueuesTest.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(hsa-queues-test ${HSA_QUEUE_TEST_DIR}/HsaQueuesTest.cpp HIPCC_OPTIONS ${HSA_QUEUE_HIPCC})
set_property(TARGET hsa-queues-test PROPERTY HIP_ARCHITECTURES OFF)
target_link_libraries(hsa-queues-test PRIVATE device device-hsa-queue-hook)
# (the link of hip_add_executable sets no run path: the library of the build tree)
target_link_options(hsa-queues-test PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:device>")

# the probes: HSA alone
foreach(probe waitlat concurrent l2damage pipemap pollstall sigdec xqueue)
    set_source_files_properties(${HSA_QUEUE_TEST_DIR}/probes/${probe}.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(hsa-queues-probe-${probe} ${HSA_QUEUE_TEST_DIR}/probes/${probe}.cpp HIPCC_OPTIONS ${HSA_QUEUE_HIPCC})
    set_property(TARGET hsa-queues-probe-${probe} PROPERTY HIP_ARCHITECTURES OFF)
    target_link_libraries(hsa-queues-probe-${probe} PRIVATE ${HIP_PATH}/lib/libhsa-runtime64.so)
endforeach()
