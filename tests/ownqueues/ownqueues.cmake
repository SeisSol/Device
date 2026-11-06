# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause

# The test of own queues and the probes that measure what an architecture takes (see README.md in
# this directory), with DEVICE_OWN_QUEUE_TESTS=ON; included by hip.cmake.

set(OWN_QUEUE_TEST_DIR ${CMAKE_CURRENT_LIST_DIR})
set(OWN_QUEUE_HIPCC -std=c++17 -O2 --offload-arch=${DEVICE_ARCH})

# the test: kernels, copies, events across queues, graphs, host functions, the mix with HIP streams,
# waits for the host; through the Device library and the launch hook
set_source_files_properties(${OWN_QUEUE_TEST_DIR}/OwnQueuesTest.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(own-queues-test ${OWN_QUEUE_TEST_DIR}/OwnQueuesTest.cpp HIPCC_OPTIONS ${OWN_QUEUE_HIPCC})
set_property(TARGET own-queues-test PROPERTY HIP_ARCHITECTURES OFF)
target_link_libraries(own-queues-test PRIVATE device device-own-queue-hook)

# the probes: HSA alone
foreach(probe waitlat concurrent l2damage pipemap pollstall sigdec xqueue)
    set_source_files_properties(${OWN_QUEUE_TEST_DIR}/probes/${probe}.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(own-queues-probe-${probe} ${OWN_QUEUE_TEST_DIR}/probes/${probe}.cpp HIPCC_OPTIONS ${OWN_QUEUE_HIPCC})
    set_property(TARGET own-queues-probe-${probe} PROPERTY HIP_ARCHITECTURES OFF)
    target_link_libraries(own-queues-probe-${probe} PRIVATE ${HIP_PATH}/lib/libhsa-runtime64.so)
endforeach()
