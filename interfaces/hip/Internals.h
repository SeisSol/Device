// SPDX-FileCopyrightText: 2020 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEISSOLDEVICE_INTERFACES_HIP_INTERNALS_H_
#define SEISSOLDEVICE_INTERFACES_HIP_INTERNALS_H_

#include "hip/hip_runtime.h"

#include <functional>
#include <initializer_list>
#include <vector>

#define APIWRAP(call) (void)::device::internals::checkResult(call, __FILE__, __LINE__, {})
#define APIWRAPX(call, except) ::device::internals::checkResult(call, __FILE__, __LINE__, except)
#define CHECK_ERR APIWRAP(hipGetLastError())

namespace device::internals {

constexpr static int DefaultBlockDim = 1024;

using DeviceStreamT = hipStream_t;

/**
 * What a stream is currently recording into: the capture status, the graph the operations end up
 * in, and the nodes a subsequently recorded operation would depend on.
 */
struct CaptureState {
  hipStreamCaptureStatus status{};
  hipGraph_t graph{nullptr};
  std::vector<hipGraphNode_t> frontier;
};

CaptureState captureState(hipStream_t stream);

/**
 * Gives the capture of the stream a node to stand on, an empty one, if it has none there yet;
 * before an event gets recorded there. HIP (seen with ROCm 6.2 and 6.3) keeps the capture state
 * that an event got when it was recorded last, within an earlier capture, if it gets recorded where
 * the capture has no node yet (e.g. right at its beginning): the streams that wait for it then
 * depend on a node of the earlier graph, never join this capture, and ending it fails ("capturing
 * stream has unjoined work"). With the empty node, the event stands for a node of this capture.
 */
void anchorCapture(hipStream_t stream);

/**
 * During a capture, HIP makes a stream that waits for an event of another capturing stream a
 * branch of that stream, also when the waiting stream takes part in the capture already. Streams
 * that wait for each other in turn thus end up as branches of each other, and ending the capture
 * then recurses through that cycle without end (seen with ROCm 7.2). Hence, an event recorded
 * during a capture remembers the nodes it stands for; a stream that takes part in the same capture
 * already adds them to its dependencies instead of waiting for the event, which is what the wait
 * does besides making it a branch.
 */
void noteRecordedEvent(hipEvent_t event, hipStream_t stream);

/**
 * Makes the stream depend on what the event stands for, if both take part in the same capture.
 * Returns whether it did; otherwise, the stream still has to wait for the event.
 */
bool dependOnCapturedEvent(hipStream_t stream, hipEvent_t event);

void forgetEvent(hipEvent_t event);

/**
 * Hands a host function to the graph that is being recorded, which keeps it alive for as long as
 * the graph can be replayed, and returns the copy to pass to the runtime. Returns nullptr if the
 * graph is not one of ours.
 */
std::function<void()>* adoptHostFunction(hipGraph_t graph, const std::function<void()>& function);
void forgetHostFunctions(hipGraph_t graph);
// Every wrapped call goes through here, so the parameters stay free of anything that allocates:
// the file name is the string literal __FILE__ expands to, and the accepted errors are read from
// the caller's temporary array.
hipError_t checkResult(hipError_t error,
                       const char* file,
                       int line,
                       std::initializer_list<hipError_t> except);
inline dim3 computeGrid1D(const dim3& block, const size_t size) {
  int numBlocks = (size + block.x - 1) / block.x;
  return dim3(numBlocks, 1, 1);
}

inline dim3 computeGrid1D(const int& leadingDim, const size_t size) {
  int numBlocks = (size + leadingDim - 1) / leadingDim;
  return dim3(numBlocks, 1, 1);
}

inline dim3 computeBlock1D(const int& leadingDim, const size_t size) {
  int numItems = ((size + leadingDim - 1) / leadingDim) * leadingDim;
  return dim3(numItems, 1, 1);
}

} // namespace device::internals

#endif // SEISSOLDEVICE_INTERFACES_HIP_INTERNALS_H_
