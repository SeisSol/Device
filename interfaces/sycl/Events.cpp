// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Internals.h"
#include "SyclWrappedAPI.h"

#include <algorithm>
#include <cassert>
#include <optional>

#ifdef ONEAPI_UNDERHOOD
#include <sycl/queue.hpp>
#endif // ONEAPI_UNDERHOOD

using namespace device;
using namespace device::internals;

namespace {
struct Event {
  std::optional<sycl::event> syclEvent;
  bool withTiming{false};
  // recorded as a profiling tag (sycl_ext_oneapi_profiling_tag), which has its time also on queues
  // without profiling
  bool profilingTag{false};
};

/// whether a graph records what gets submitted to the queue
[[maybe_unused]] bool recording([[maybe_unused]] const sycl::queue& queue) {
#ifdef SYCL_EXT_ONEAPI_GRAPH
  return queue.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;
#else
  return false;
#endif
}
} // namespace

void* ConcreteAPI::createEvent(bool withTiming) {
  return new Event{std::nullopt, withTiming, false};
}

bool ConcreteAPI::isCapableOfEventTiming() {
#ifdef PROFILING_ENABLED
  return true;
#else
  return this->currentContext()->profilingTags;
#endif
}

double ConcreteAPI::timespanEvents(void* eventPtrStart, void* eventPtrEnd) {
  auto* start = static_cast<Event*>(eventPtrStart);
  auto* end = static_cast<Event*>(eventPtrEnd);

#ifndef PROFILING_ENABLED
  // Profiling information is only there when the queue carries the enable_profiling property, or
  // for profiling tags; asking another event throws from deep inside the runtime.
  if (!(start->profilingTag && end->profilingTag)) {
    logError() << "The SYCL backend can only time events when it is built with "
                  "ENABLE_PROFILING_MARKERS=ON, or on devices with profiling tags "
                  "(sycl_ext_oneapi_profiling_tag) for events recorded outside of graphs.";
    return 0.0;
  }
#endif

  if (!(start->withTiming && end->withTiming)) {
    logError() << "Timing was not requested for at least one of the events given for timing "
                  "calculation.";
  }

  if (!(start->syclEvent.has_value() && end->syclEvent.has_value())) {
    logError() << "Invalid events given for timing calculation.";
  }

  const auto startTime =
      start->syclEvent.value().get_profiling_info<sycl::info::event_profiling::command_end>();
  const auto endTime =
      end->syclEvent.value().get_profiling_info<sycl::info::event_profiling::command_end>();

  // cf. https://oneapi-src.github.io/SYCLomatic/dev_guide/reference/diagnostic_ref/dpct1012.html

  return static_cast<double>(endTime - startTime) / 1'000'000'000.0;
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  delete event;
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  if (event->syclEvent) {
    waitCheck(event->syclEvent.value());
  }
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  // (NOTE: we do not poll here on the SYCL implementation, i.e. we may get MPI-like issues)
  return !event->syclEvent.has_value() ||
         event->syclEvent.value().get_info<sycl::info::event::command_execution_status>() ==
             sycl::info::event_command_status::complete;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  event->syclEvent = std::make_optional<sycl::event>();
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  auto* queue = static_cast<sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

#if !defined(PROFILING_ENABLED) && defined(SYCL_EXT_ONEAPI_PROFILING_TAG)
  // a tag completes once all work submitted before it has; on an in-order queue, it is a light
  // marker
  event->profilingTag =
      event->withTiming && this->currentContext()->profilingTags && !recording(*queue);
  if (event->profilingTag) {
    event->syclEvent = sycl::ext::oneapi::experimental::submit_profiling_tag(*queue);
    return;
  }
#endif

  event->syclEvent = std::make_optional<sycl::event>(
      queue->submit([&](sycl::handler& h) { DEVICE_SYCL_EMPTY_OPERATION(h); }));
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  auto* queue = static_cast<sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

  if (!event->syclEvent.has_value()) {
    logError() << "Event not yet recorded; no synchronization possible.";
  }

  queue->submit([&](sycl::handler& h) {
    DEVICE_SYCL_EMPTY_OPERATION_WITH_EVENT(h, event->syclEvent.value());
  });
}
