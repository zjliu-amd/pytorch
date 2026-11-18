#include <gtest/gtest.h>

#include <ATen/DeviceAccelerator.h>
#include <ATen/cuda/CUDAContext.h>


TEST(CudaExchangeDeviceTest, checkPrimaryContext) {
  if (!at::cuda::is_available()) {
    return;
  }

  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::cuda::MaybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::accelerator::maybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));

  if (at::cuda::device_count() > 1) {
    ASSERT_FALSE(at::cuda::hasPrimaryContext(1));
    at::cuda::ExchangeDevice(1);
    ASSERT_TRUE(at::cuda::hasPrimaryContext(1));
  }

  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::cuda::MaybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::accelerator::maybeExchangeDevice(0);
  ASSERT_FALSE(at::cuda::hasPrimaryContext(0));
  at::accelerator::exchangeDevice(0);
  if (at::cuda::device_count() == 1) {
    // With one device, device 0 has been current from the start, so
    // exchangeDevice(0) returns without calling cudaSetDevice/hipSetDevice
    // (c10::cuda::ExchangeDevice) and nothing above creates the context. The
    // check below needs a second device to switch away from and back.
    GTEST_SKIP() << "checking that setting the device creates its primary "
                    "context needs at least two devices";
  }
  ASSERT_TRUE(at::cuda::hasPrimaryContext(0));
}
