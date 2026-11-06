<!--
    SPDX-FileCopyrightText: 2026 SeisSol Group

    SPDX-License-Identifier: BSD-3-Clause
-->

# Tests and probes of own queues

Built with `-DDEVICE_BACKEND=hip -DDEVICE_OWN_QUEUE_TESTS=ON` (and own queues,
the default on AMD GPUs, see `interfaces/hip/ownqueues/OwnQueues.h`). Run them
on a GPU of their own: they keep several queues busy at once.

`own-queues-test` runs the own queues through the Device library and the launch
hook: kernels and copies, events across queues, own streams and streams of HIP
waiting for each other, a graph with a temporary allocation and branches, host
functions, 200000 launches, launches behind a wait for the host, counters; then
the host and device time of launches against HIP. It exits with 1 if a check
fails.

The probes `own-queues-probe-*` use HSA alone. They measure what the table of
architectures (`interfaces/hip/ownqueues/Arch.cpp`) says about a new one:

- `waitlat`: the latency of a wait for another queue, by kind, and whether the
  command processor takes barrier-value packets (else INVALID_PACKET_FORMAT):
  `barrierValue`.
- `pipemap`: whether queues share the pipes of the command processor, and
  whether the pipes stay put: `spreadPipes`.
- `concurrent`: whether kernels without the barrier bit run at the same time
  on one queue (the lanes of graphs).
- `pollstall`: whether queues that wait slow down the kernels of another one
  (the kind of wait).
- `sigdec`: whether signals that several queues count down lose decrements
  (one counter per queue).
- `l2damage`: what fences of the system cost the kernels of other queues (the
  scopes of the fences).
- `xqueue`: the latencies of chains, ping-pongs and waits for the host, against
  HIP (an overview).

The time of kernel arguments in device memory (`deviceKernargs`) shows in the
last part of `own-queues-test` (large kernels) and in programs.
