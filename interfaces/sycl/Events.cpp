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
  // not known, if possible with SYCL at the moment
  syncEventWithHost(eventPtr);
}

void ConcreteAPI::recordEventOnHost(void* eventPtr) {
  auto* event = static_cast<Event*>(eventPtr);
  event->syclEvent = std::make_optional<cl::sycl::event>();
}

void ConcreteAPI::recordEventOnStream(void* eventPtr, void* streamPtr) {
  auto* queue = static_cast<cl::sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

#ifdef SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
  event->syclEvent = std::make_optional<cl::sycl::event>(queue->ext_oneapi_submit_barrier());
#else
  event->syclEvent = std::make_optional<cl::sycl::event>(queue->submit([&](cl::sycl::handler& h) {
    h.single_task([=](){});
  }));
#endif
}

void ConcreteAPI::syncStreamWithEvent(void* streamPtr, void* eventPtr) {
  auto* queue = static_cast<cl::sycl::queue*>(streamPtr);
  auto* event = static_cast<Event*>(eventPtr);

#ifdef SYCL_EXT_ONEAPI_ENQUEUE_BARRIER
  queue->ext_oneapi_submit_barrier({event->syclEvent.value()});
#else
  queue->submit([&](cl::sycl::handler& h) {
    h.depends_on(event->syclEvent.value());
    h.single_task([=](){});
  });
#endif
}
