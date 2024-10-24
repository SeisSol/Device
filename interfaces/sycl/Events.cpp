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
    std::optional<cl::sycl::event> syclEvent;
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
  return event->get_info<cl::sycl::info::event::command_execution_status>() == cl::sycl::info::event_command_status::complete;
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  event->syclEvent = std::make_optional<cl::sycl::event>();
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  auto* queue = static_cast<cl::sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

  event->syclEvent = std::make_optional<cl::sycl::event>(queue->submit([&](cl::sycl::handler& h) {
    DEVICE_SYCL_EMPTY_OPERATION(h);
  }));
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  auto* queue = static_cast<cl::sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

  queue->submit([&](cl::sycl::handler& h) {
    h.depends_on(event->syclEvent.value());
    DEVICE_SYCL_EMPTY_OPERATION(h);
  });
}
