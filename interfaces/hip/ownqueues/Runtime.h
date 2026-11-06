// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_RUNTIME_H_
#define SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_RUNTIME_H_

#include "Arch.h"
#include "Common.h"
#include "HostThread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace device::ownqueues {

class Queue;

class Runtime {
  public:
  static Runtime& instance();

  Runtime();
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  bool setUp();
  const KernelInfo& kernel(const void* function);
  Queue& queue(int number);
  std::vector<Queue*> allQueues();
  void finalize();

  /// the short name of a kernel object, for the trace
  std::string kernelName(std::uint64_t object);
  /// the number of the queue whose marks count the signal down, or nothing
  std::optional<int> queueOfCounter(std::uint64_t signal);

  // the trace (see trace()): whether the queues give their packets signals that take the times,
  // from a pool
  std::atomic<bool> tracing{false};
  hsa_signal_t traceSignal();
  void freeTraceSignal(hsa_signal_t signal);

  hsa_agent_t gpu{};
  hsa_agent_t cpu{};
  hsa_amd_memory_pool_t kernargPool{};
  bool havePool{false};
  // kernel arguments in device memory that the host writes (DEVICE_OWN_QUEUE_KERNARGS=device, the
  // default on MI250X and MI300), instead of host memory
  hsa_amd_memory_pool_t deviceKernargPool{};
  bool haveDevicePool{false};
  bool deviceKernargs{false};
  // the fences of marks and of waits for other queues (DEVICE_OWN_QUEUE_MARK_SCOPE, system by
  // default, and DEVICE_OWN_QUEUE_WAIT_SCOPE, agent by default: system or agent)
  std::uint16_t markFences{MarkFences};
  std::uint16_t waitFences{QueueWaitFences};
  // how the work after a wait for the host, and copies from host memory, see what it wrote: by a
  // kernel that fences in the shader (DEVICE_OWN_QUEUE_HOST_ACQUIRE=kernel, the default, see
  // ownAcquireSystem), or by the acquire fence of the packet (=packet)
  bool acquireByKernel{true};
  // events time what they enclose (DEVICE_OWN_QUEUE_TIMING=1): by a kernel that takes the time
  bool timing{false};
  // waits between own streams and streams of HIP (e.g. those of NCCL) on the device
  // (DEVICE_OWN_QUEUE_HIP_WAITS=device, the default; =host waits on the host): a stream of HIP
  // waits for a mark by a kernel of ours (ownHipWaitBelow), an own stream for an event of HIP by a
  // counter that a kernel of ours sets on that stream after each event (see recordEvent())
  bool hipWaitsOnDevice{true};
  // DEVICE_OWN_QUEUE_SCHEDULE=1: graphs spread the kernels of their levels over own queues on other
  // pipes where that ends a level earlier, by the durations they learn and the predicted work of
  // the queues (see Program::submitScheduled()); a level gets spread only with at least
  // DEVICE_OWN_QUEUE_SCHEDULE_MIN_US of work, with forks and joins that take ..._FORK_US each
  bool schedule{false};
  double scheduleMinWork{120e-6};
  double forkSeconds{5e-6};
  // a level gets spread only if none of its kernels has more workgroups than this
  // (DEVICE_OWN_QUEUE_SCHEDULE_MAX_BLOCKS, the compute units of the device by default): larger ones
  // fill the device by themselves, and spreading them made a mesh of 1.4 million cells slower
  std::uint64_t scheduleMaxBlocks{128};
  /// the duration of a kernel (by its kernel object) the graphs have measured, or 0
  double kernelEstimate(std::uint64_t object);
  void learnKernel(std::uint64_t object, double seconds);
  // DEVICE_OWN_QUEUE_PIPES=check: each new queue gets checked for the pipes it shares; =spread: own
  // queues get pipes of their own as long as there are (see placeQueue())
  bool pipeCheck{false};
  bool pipeSpread{false};
  // own queues that are idle most of the time, which may share a pipe with another one
  // (DEVICE_OWN_QUEUE_LIGHT, a list of queue numbers)
  std::set<int> lightQueues;
  /**
   * An HSA queue for the own queue `number`, on a pipe with as few other own queues as possible.
   * The hardware scheduler gives the i-th queue of a process the pipe i modulo their number (the
   * MI250X: 4 pipes for the processes): queues that get created later take the pipes in turn, and
   * those after a queue that goes move up by one. So the first call finds the pipes, with an idle
   * queue on each (until a new queue is on a known one), and each call creates queues until one is
   * on a pipe as little busy as any (by its own queues that are not light, see lightQueues); the
   * others stay unused, as do the idle ones: destroyed, they would move the queues after them.
   */
  hsa_queue_t* placeQueue(int number);
  /// the own queues that the graphs of `self` may lend lanes from: on other pipes (if known), alone
  /// there but for light queues, not light themselves, no helpers
  std::vector<Queue*> borrowable(const Queue& self);
  std::uint64_t timestampFrequency{1};
  // the system timestamp when the own queues started; stamps before it are not valid
  std::uint64_t startTimestamp{0};
  hsa_ven_amd_loader_1_03_pfn_t loader{};
  // whether the command processor waits for signal values itself (barrier-value packets)
  bool barrierValue{false};
  std::uint32_t stackSize{1024};
  std::uint32_t queuePackets{1U << 14};
  std::uint64_t kernargRingBytes{DefaultKernargRingMiB << 20};
  std::uint32_t bus{0};
  std::uint32_t slot{0};
  std::uint32_t domain{0};
  // the compute units that the waves of a queue may use, by queue number
  // (DEVICE_OWN_QUEUE_CU_MASKS, e.g. "-10:0-23;1:24-109"); all of them for the others
  std::map<int, std::vector<std::uint32_t>> computeUnitMasks;
  std::uint32_t computeUnits{0};
  HostThread hostThread;

  // the symbols of the kernels, from the loaded code objects; by their names (with .kd)
  std::unordered_map<std::string, KernelInfo> symbols;
  std::unordered_set<std::uint64_t> parsedCodeObjects;

  private:
  void scanCodeObjects();
  /// writes what the own queues use into the log
  void describe();

  std::mutex mutex_;
  std::unordered_map<const void*, const KernelInfo*> kernels_;
  std::map<int, std::unique_ptr<Queue>> queues_;
  // what the table of architectures says about the device (see Arch.h), and whether the log has the
  // description already: it gets written with the first queue, not when set up, as a program may
  // tell the logger its rank only later (SeisSol sets up the device before MPI_Init), and every
  // process would write it then
  ArchTraits arch_;
  bool described_{false};
  // the pipes found so far (see placeQueue()): an idle queue on each, and the own queues on it
  struct Pipe {
    hsa_queue_t* probe{nullptr};
    std::vector<int> queues;
  };
  std::mutex estimateMutex_;
  std::unordered_map<std::uint64_t, double> kernelEstimates_;
  std::mutex placeMutex_;
  std::vector<Pipe> pipes_;
  bool pipesKnown_{false};
  // queues that stay unused: destroyed, they could move the other queues to other pipes
  std::vector<hsa_queue_t*> spareQueues_;
  std::unordered_map<std::uint64_t, std::string> kernelNames_;
  std::mutex traceMutex_;
  std::vector<hsa_signal_t> traceSignals_;
};

/// memory for kernel arguments: in device memory that the host writes, or in host memory
void* allocKernargMemory(std::size_t bytes);
/// after the host has written kernel arguments into device memory: they are there before the
/// packet that reads them gets published
void flushKernargs(const void* last);
hsa_queue_t* createHsaQueue();
/// reads the kernels of the code objects of an executable into Runtime::symbols (see
/// Metadata.cpp)
hsa_status_t visitExecutable(hsa_executable_t executable, void* data);

} // namespace device::ownqueues

#endif // SEISSOLDEVICE_INTERFACES_HIP_OWNQUEUES_RUNTIME_H_
