import functools


@functools.cache
def has_helion_package() -> bool:
    try:
        import helion  # type: ignore[import-untyped, import-not-found]  # noqa: F401
    except ImportError:
        return False
    except Exception as e:
        # helion is an optional dependency; a broken install (e.g. a release that
        # fails at import time on this Python version) must not crash every
        # module that probes for it, so report it as unavailable instead.
        import warnings

        warnings.warn(
            f"helion is installed but failed to import ({type(e).__name__}: {e}); "
            "treating it as unavailable"
        )
        return False
    return True


@functools.cache
def has_helion() -> bool:
    return has_helion_package()
