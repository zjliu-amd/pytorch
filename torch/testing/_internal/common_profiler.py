from torch.profiler import ProfilerActivity, supported_activities


def device_tracing_supported(device_type) -> bool:
    """Whether torch.profiler can trace activity on ``device_type`` (kernels,
    memcpy/memset, runtime API calls, CPU-to-device flows).

    ``kineto_available()`` alone is not enough: a Kineto build without a device
    tracing backend (e.g. without CUPTI or roctracer) records CPU activity only,
    and ``supported_activities()`` then omits the device activity.
    """
    activity = getattr(ProfilerActivity, str(device_type).upper(), None)
    return activity is not None and activity in supported_activities()


DEVICE_TRACING_UNSUPPORTED_MSG = (
    "torch.profiler cannot trace device activity in this build "
    "(the device activity is not in torch.profiler.supported_activities())"
)
