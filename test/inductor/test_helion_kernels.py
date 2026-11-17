# Owner(s): ["module: inductor"]
import importlib.abc
import importlib.util
import sys
from unittest import mock

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_HELION,
    requires_helion,
    requires_triton,
)
from torch.utils._helion import has_helion, has_helion_package


if HAS_HELION:
    import helion
    import helion.language as hl


class _BrokenHelionLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ValueError("simulated broken helion install")


class _BrokenHelionFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "helion":
            return importlib.util.spec_from_loader(fullname, _BrokenHelionLoader())
        return None


class HelionProbeTests(TestCase):
    def test_has_helion_broken_install(self):
        # An installed helion that raises a non-ImportError at import time must
        # be reported as unavailable (with a warning), not crash the caller.
        def clear_caches():
            has_helion_package.cache_clear()
            has_helion.cache_clear()

        clear_caches()
        self.addCleanup(clear_caches)
        meta_path = [_BrokenHelionFinder(), *sys.meta_path]
        with (
            mock.patch.dict(sys.modules),
            mock.patch.object(sys, "meta_path", meta_path),
        ):
            sys.modules.pop("helion", None)
            with self.assertWarnsRegex(UserWarning, "simulated broken helion install"):
                self.assertFalse(has_helion())


class HelionTests(TestCase):
    @requires_triton()
    @requires_helion()
    def test_add_kernel(self):
        @helion.kernel(config=helion.Config(block_sizes=[1, 2]))
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # match pytorch broadcasting rules
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty(
                x.shape,
                # match type promotion of torch.add
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            # tile will be a tuple of blocks
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return add(x, y)

        x = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)

        out = add(x, y)
        compiled_add = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(compiled_out, x + y)

    @requires_triton()
    @requires_helion()
    def test_softmax_view_reshape(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).view(tile_n, 1)
                exp = torch.exp(values - amax)
                sum_exp = torch.reshape(torch.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=GPU_TYPE, dtype=torch.float16)
        result = softmax(x)
        self.assertEqual(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )


instantiate_parametrized_tests(HelionTests)


if __name__ == "__main__":
    run_tests()
