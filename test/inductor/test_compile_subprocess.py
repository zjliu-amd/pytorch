# Owner(s): ["module: fx"]

#
# Tests compiling the inductor tests in a subprocess.
#

import contextlib
import importlib
import io
import logging
import os
import sys
import time
import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch.library
from torch._inductor.compile_fx import _InProcessFxCompile, FxCompile, FxCompileMode
from torch._inductor.compile_fx_subproc import _SubprocessFxCompile
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    isRocmArchAnyOf,
    MI350_ARCH,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    IS_BIG_GPU,
    requires_gpu,
    requires_triton,
    RUN_CPU,
    RUN_GPU,
)
from torch.testing._internal.logging_utils import logs_to_string


if IS_WINDOWS and IS_CI:
    # TODO(xuhancn) : Debug and confirm pass_fds status on Windows.
    sys.stderr.write(
        "Almost UTs failed: pass_fds not supported on Windows, skip them on Windows.\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("pass_fds not supported on Windows")


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import inductor.test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    # TypeError: cannot pickle 'generator' object
    "test_layer_norm": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_remove_noop_slice": TestFailure(
        ("cuda",),
        is_skip=(TEST_WITH_ROCM and isRocmArchAnyOf(MI350_ARCH)) or not TEST_WITH_ROCM,
    ),
    "test_remove_noop_slice1": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_slice_scatter": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_view_default": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_view_dtype": TestFailure(("xpu"), is_skip=True),
    # can not pickle ParametrizedConv2d
    "test_weight_norm_conv2d": TestFailure(("cpu", "cuda"), is_skip=True),
    # TypeError: cannot pickle '_thread._local' object.
    # nested_compile_region(options=...) hangs compiler callables off
    # NestedCompileRegionOptions. dill cannot resolve a module reference for that
    # class, falls back to pickling the callables by value, and walking their
    # globals reaches a thread-local. is_skip rather than xfail because the
    # by-value fallback is dill-specific, so the test still passes where the
    # graph pickler does not go through dill.
    "test_regional_fallback_by_default_invoke_subgraph": TestFailure(
        ("cpu", "cuda"), is_skip=True
    ),
    # This manually constructs an FX graph with an OpOverloadPacket target to
    # cover a legacy lowering table entry, which is outside the subprocess
    # compile serialization path this file exercises.
    "test_split_overload_packet_lowering": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),
}
# Parametrization changes the attribute keys but preserves each template name.
# Expand to the generated keys that copy_tests uses to look up failures.
_subprocess_skips = (
    "test_conv2d_backward_parametrized",
    "test_conv2d_backward_input_layout",
    "test_conv_transpose2d_forward_triton",
)
test_failures.update(
    {
        name: TestFailure(("cpu", "cuda", "xpu", "mps"), is_skip=True)
        for name, fn in vars(inductor.test_torchinductor.CommonTemplate).items()
        if getattr(fn, "__name__", None) in _subprocess_skips
    }
)


class TestSubprocess(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        FxCompile._reset_stats()

        super().setUp()

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch(
                "torch._inductor.compile_fx.fx_compile_mode",
                FxCompileMode.SUBPROCESS,
            )
        )

    def tearDown(self):
        # Check that the test didn't instigate an in-process compile - which
        # would mean that something about the fx graph failed to serialize. If
        # some tests are expected to fail then we should probably add a list of
        # expected failures here.
        self.assertEqual(
            FxCompile._compile_stats[type(_InProcessFxCompile)].codegen_and_compile, 0
        )
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()

    @requires_gpu()
    @requires_triton()
    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @patch("torch._inductor.compile_fx.fx_compile_progressive", True)
    def test_progressive(self):
        from torch._inductor.compile_fx_async import _ProgressiveFxCompile

        x = torch.randn(1152, 4096, device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn(4096, 4096, device=GPU_TYPE, dtype=torch.bfloat16)

        @torch.compile(fullgraph=True, backend="inductor")
        def optimized(x, y):
            return (x @ y).relu()

        _ProgressiveFxCompile._reset_stats()

        source_codes: list[str] = []

        def save_output_code(code: str) -> None:
            source_codes.append(code)

        with contextlib.ExitStack() as stack:
            # When this bug is fixed, remove the cache disabling below
            if not torch._inductor.compile_fx_async.BUG_CACHES_DONT_WORK_WITH_ASYNC:
                raise AssertionError
            stack.enter_context(
                torch._inductor.config.patch(
                    autotune_local_cache=False, fx_graph_cache=False
                )
            )
            stack.enter_context(
                mock.patch.object(GraphLowering, "save_output_code", save_output_code)
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )

            # How long to wait (in seconds) before giving up.
            TIMEOUT = 300
            # If non-None then how often (in seconds) to print a TICK message.
            TICK_REPORT = None

            start = time.time()
            last_report = start
            while _ProgressiveFxCompile._stat_optimized_runs < 4:
                time.sleep(0.25)

                optimized(x, y)

                now = time.time()
                if TICK_REPORT is not None and (now - last_report > TICK_REPORT):
                    print(f"*** TICK {int(now - start)}")
                    last_report = now

                if now - start > TIMEOUT:
                    raise RuntimeError(
                        "Test timed out before producing a progressively optimized compiled artifact."
                    )

            self.assertEqual(_ProgressiveFxCompile._stat_optimized_runs, 4)
            self.assertGreater(_ProgressiveFxCompile._stat_fast_runs, 0)
            self.assertGreaterEqual(_ProgressiveFxCompile._stat_bg_started, 1)
            self.assertGreaterEqual(_ProgressiveFxCompile._stat_bg_finished, 1)

        self.assertIn("'max_autotune': True", source_codes[-1])

    @patch("torch._inductor.compile_fx.fx_compile_async", True)
    def test_async(self):
        # Test that async+subprocess works.
        from torch._inductor.compile_fx_async import _AsyncFxCompile

        @torch.compile(fullgraph=True, backend="inductor")
        def model_add(x, y):
            out = x
            for _ in range(500):
                out = torch.add(out, y)
            return out

        _AsyncFxCompile._reset_stats()

        with contextlib.ExitStack() as stack:
            if not torch._inductor.compile_fx_async.BUG_CACHES_DONT_WORK_WITH_ASYNC:
                raise AssertionError
            stack.enter_context(
                torch._inductor.config.patch(
                    autotune_local_cache=False, fx_graph_cache=False
                )
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )

            # How long to wait (in seconds) before giving up.
            TIMEOUT = 300
            # If non-None then how often (in seconds) to print a TICK message.
            TICK_REPORT = None

            start = time.time()
            last_report = start
            while True:
                start_stat_compiled_runs = _AsyncFxCompile._stat_compiled_runs
                # Sleep a bit so we don't drive the CPU unnecessarily.
                time.sleep(0.25)

                x = torch.randn(100, 100, requires_grad=True)
                y = torch.randn(100, 100, requires_grad=True)

                # Forward pass
                output = model_add(x, y)

                # Backward pass
                output.sum().backward()

                if _AsyncFxCompile._stat_compiled_runs - start_stat_compiled_runs == 2:
                    break

                # DEBUGGING: Print a periodic message so we know we're still
                # running...
                now = time.time()
                if TICK_REPORT is not None and (now - last_report > TICK_REPORT):
                    print(f"*** TICK {int(now - start)}")
                    last_report = now

                if now - start > TIMEOUT:
                    raise RuntimeError(
                        "Test timed out before producing a compiled artifact."
                    )

            self.assertGreater(_AsyncFxCompile._stat_compiled_runs, 1)
            # Make sure we ran eager at least once. Normally this will be
            # something like 80.
            self.assertGreater(_AsyncFxCompile._stat_eager_runs, 0)
            self.assertEqual(_AsyncFxCompile._stat_bg_started, 2)
            self.assertEqual(_AsyncFxCompile._stat_bg_finished, 2)


# Kept separate from TestSubprocess (the base of CpuTests/GPUTests) since these
# tests don't depend on the device so there's no point in running them for each.
@patch("torch._inductor.compile_fx.fx_compile_mode", FxCompileMode.SUBPROCESS)
class TestSubprocessLogging(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        FxCompile._reset_stats()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    @staticmethod
    def _restart_subprocess_pool():
        if _SubprocessFxCompile.process_pool.cache_info().currsize:
            _SubprocessFxCompile.process_pool().shutdown()
            _SubprocessFxCompile.process_pool.cache_clear()

    def _check_post_grad_graph_logs(self, start_worker_with_logs):
        # The child's logging config is fixed when the compile subprocess
        # starts. Changes to the logging config in the parent after that must
        # still be honored by the child: post_grad_graphs is logged exactly
        # once while it's enabled and not at all once it's disabled again.
        def fn(x, y):
            return (x + y).sin()

        def compile_and_run():
            torch._dynamo.reset()
            torch.compile(fn, backend="inductor")(torch.randn(8), torch.randn(8))

        with contextlib.ExitStack() as stack:
            stack.enter_context(torch._inductor.config.patch(fx_graph_cache=False))
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )

            # Make sure the compile subprocess is started at a known point.
            self._restart_subprocess_pool()
            stack.callback(self._restart_subprocess_pool)
            if not start_worker_with_logs:
                compile_and_run()

            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                compile_and_run()
            self.assertEqual(log_stream.getvalue().count("AFTER POST GRAD"), 1)

            artifact_logger = torch._logging.getArtifactLogger(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            artifact_logger.addHandler(handler)
            stack.callback(artifact_logger.removeHandler, handler)
            compile_and_run()
            self.assertNotIn("AFTER POST GRAD", log_stream.getvalue())

        self.assertEqual(
            FxCompile._compile_stats[_SubprocessFxCompile].codegen_and_compile,
            2 if start_worker_with_logs else 3,
        )

    def test_artifact_logs_enabled_after_worker_start(self):
        self._check_post_grad_graph_logs(start_worker_with_logs=False)

    def test_artifact_logs_disabled_after_worker_start(self):
        self._check_post_grad_graph_logs(start_worker_with_logs=True)


if RUN_CPU:

    class CpuTests(TestSubprocess):
        common = check_model
        device = "cpu"

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, CpuTests, "cpu", test_failures
    )

if RUN_GPU and not TEST_WITH_ASAN:

    class GPUTests(TestSubprocess):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, GPUTests, GPU_TYPE, test_failures
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU or RUN_GPU:
        run_tests(needs="filelock")
