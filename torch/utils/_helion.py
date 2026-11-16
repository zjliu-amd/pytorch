import functools
import warnings


@functools.cache
def has_helion_package() -> bool:
    try:
        import helion  # type: ignore[import-untyped, import-not-found]  # noqa: F401
    except ImportError:
        return False
    except Exception as e:
        # A helion install that is present but broken (e.g. raises at import
        # time on this Python version) should not take down unrelated code
        # that merely probes for helion; treat it as unavailable, visibly.
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
