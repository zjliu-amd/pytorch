#include <ATen/core/CachingHostAllocator.h>

#include <array>

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#endif

namespace at {

namespace {

std::array<HostAllocator*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_array{};
std::array<uint8_t, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_priority{};

} // anonymous namespace

#ifdef _WIN32
namespace detail {

void* open_current_thread_handle() {
  // GetCurrentThread() returns a pseudo-handle that always means "the calling
  // thread", so open a real one.
  return OpenThread(SYNCHRONIZE, FALSE, GetCurrentThreadId());
}

bool has_thread_exited(void* thread_handle) {
  return WaitForSingleObject(static_cast<HANDLE>(thread_handle), 0) ==
      WAIT_OBJECT_0;
}

} // namespace detail
#endif

void setHostAllocator(
    at::DeviceType device_type,
    at::HostAllocator* allocator,
    uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(device_type)]) {
    allocator_array[static_cast<int>(device_type)] = allocator;
    allocator_priority[static_cast<int>(device_type)] = priority;
  }
}

at::HostAllocator* getHostAllocator(at::DeviceType device_type) {
  auto* allocator = allocator_array[static_cast<int>(device_type)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocator, "Host Allocator for ", device_type, " is not set.");
  return allocator;
}

} // namespace at
