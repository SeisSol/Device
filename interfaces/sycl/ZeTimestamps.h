// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_SYCL_ZETIMESTAMPS_H_
#define SEISSOLDEVICE_INTERFACES_SYCL_ZETIMESTAMPS_H_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sycl/sycl.hpp>
#include <vector>

namespace device {

/*
 * Timestamps of the device for the events with timing, taken through Level Zero when SYCL runs on
 * it (built with DEVICE_SYCL_LEVEL_ZERO): a barrier and a write of the global timestamp of the
 * device into host memory, appended to the immediate command list of the in-order queue.
 *
 * Profiling tags (sycl_ext_oneapi_profiling_tag) take the same timestamp through the runtime, but
 * with oneAPI 2025.0 on a Max 1550, a tag took about 160 us on the host, and the device about 300
 * us longer; a timestamp here takes about 1 us on the host and a few on the device.
 *
 * SYCL knows nothing of these commands: a wait for the queue does not cover them. An event of
 * Level Zero, which the write signals, tells when the timestamp is there. (A SYCL queue cannot wait
 * for such an event: oneAPI 2025.0 runs work that depends on an interop event right away.)
 */
class ZeTimestamps {
  public:
  /// a timestamp in host memory, and the event of Level Zero that tells when it is there
  struct Stamp {
    std::uint64_t* value{nullptr};
    void* event{nullptr}; // ze_event_handle_t
    // whether a write of the timestamp was appended (and may still be on its way)
    bool recorded{false};
  };

  ZeTimestamps() = default;
  ~ZeTimestamps();
  ZeTimestamps(const ZeTimestamps&) = delete;
  ZeTimestamps& operator=(const ZeTimestamps&) = delete;
  ZeTimestamps(ZeTimestamps&&) = delete;
  ZeTimestamps& operator=(ZeTimestamps&&) = delete;

  /*
   * Whether timestamps can be taken on the queues of this device: Level Zero underneath, and
   * queues with immediate command lists (as `queue`, e.g. the default queue).
   */
  bool possible(sycl::queue& queue);

  /*
   * Appends a timestamp to the queue (reusing `stamp` if it has one already, once it is there).
   * Returns false if that is not possible here; then nothing happened.
   */
  bool record(sycl::queue& queue, Stamp& stamp);

  /*
   * Takes the timestamp of the device now, as for a queue without work. Returns false if the
   * timestamps were not set up yet (by a record on a queue).
   */
  bool recordNow(Stamp& stamp);

  [[nodiscard]] bool completed(const Stamp& stamp) const;
  void wait(const Stamp& stamp) const;

  /// the time from `start` to `end`, in seconds; negative if `end` was first
  [[nodiscard]] double seconds(const Stamp& start, const Stamp& end) const;

  /// gives the timestamp back for reuse
  void release(Stamp& stamp);

  private:
  bool setUp(sycl::queue& queue);
  bool acquire(Stamp& stamp);

  std::mutex mutex;
  [[maybe_unused]] bool tried{false};
  [[maybe_unused]] bool usable{false};
  [[maybe_unused]] void* context{nullptr}; // ze_context_handle_t
  [[maybe_unused]] void* device{nullptr};  // ze_device_handle_t
  // the length of a tick: as the device reports it at first, then as measured (see calibrate())
  std::atomic<double> nanosecondsPerTick{0};
  std::uint64_t validMask{~std::uint64_t{0}};
  /*
   * Measures the length of a tick against the steady clock of the host, from two readings of both
   * clocks (zeDeviceGetGlobalTimestamps) at least a second apart. A Max 1550 reports 80 ns, but
   * ticks every 80.11 ns: the times of the device would run 0.135 % short, and a trace that sets
   * them against the host would drift by 1.35 ms per second. (With the mutex held.)
   */
  void calibrate();
  // the reading of both clocks the length gets measured from (the host in ns of the steady clock),
  // and when the length was measured last
  [[maybe_unused]] std::uint64_t calibrationHost{0};
  [[maybe_unused]] std::uint64_t calibrationDevice{0};
  [[maybe_unused]] std::uint64_t lastCalibration{0};
  std::optional<sycl::context> syclContext{};
  std::vector<void*> pools;           // ze_event_pool_handle_t
  std::vector<std::uint64_t*> values; // host memory, a block per pool
  std::vector<Stamp> available;
  std::vector<Stamp> all;
};

} // namespace device

#endif // SEISSOLDEVICE_INTERFACES_SYCL_ZETIMESTAMPS_H_
