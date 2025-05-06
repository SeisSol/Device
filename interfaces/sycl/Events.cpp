// SPDX-FileCopyrightText: 2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "SyclWrappedAPI.h"

#include <algorithm>
#include <cassert>
#include <optional>

#ifdef ONEAPI_UNDERHOOD
#include <sycl/queue.hpp>
#endif // ONEAPI_UNDERHOOD

using namespace device;

namespace {
struct Event {
    std::optional<sycl::event> syclEvent;
};
} // namespace

void* ConcreteAPI::createEvent() {
  return new Event();
}

void ConcreteAPI::destroyEvent(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  delete event;
}

void ConcreteAPI::syncEventWithHost(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  if (event->syclEvent) {
    event->syclEvent.value().wait_and_throw();
  }
}

bool ConcreteAPI::isEventCompleted(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  // (NOTE: we do not poll here on the SYCL implementation, i.e. we may get MPI-like issues)
  return !event->syclEvent.has_value() || event->syclEvent.value().get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  event->syclEvent = std::make_optional<sycl::event>();
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  auto* queue = static_cast<sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

  event->syclEvent = std::make_optional<sycl::event>(queue->submit([&](sycl::handler& h) {
    DEVICE_SYCL_EMPTY_OPERATION(h);
  }));
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  auto* queue = static_cast<sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

  queue->submit([&](sycl::handler& h) {
    DEVICE_SYCL_EMPTY_OPERATION_WITH_EVENT(h, event->syclEvent.value());
  });
}

