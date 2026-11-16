import functools
import warnings


@functools.cache
def has_helion_package() -> bool:
    try:
        import helion  # type: ignore[import-untyped, import-not-found]  # noqa: F401
    except ImportError:
        return False
    except Exception as e:
        # helion is an optional dependency probed at import time by test and
        # inductor utilities; a broken install must not take those down with it.
        warnings.warn(
            f"helion is installed but failed to import ({type(e).__name__}: {e}); "
            "treating helion as unavailable",
            stacklevel=2,
        )
        return False
    return True


@functools.cache
def has_helion() -> bool:
    return has_helion_package()
