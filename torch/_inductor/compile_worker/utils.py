import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from threading import Thread
from time import monotonic, sleep
from typing import Any, IO

from torch._inductor.compile_worker import watchdog


_IN_TOPLEVEL_PROCESS = True


def popen_pass_fds(fds: Sequence[int]) -> tuple[list[int], dict[str, Any]]:
    """
    Make the pipe fds `fds` available to a subprocess.Popen child.

    Returns the integers to give the child on its command line (to be turned
    back into fds with fd_from_parent) and the extra Popen kwargs. POSIX uses
    pass_fds. Windows has no pass_fds, so the child inherits the underlying OS
    handles instead, and handle_list limits inheritance to just these handles.
    """
    if sys.platform != "win32":
        return list(fds), {"pass_fds": tuple(fds)}
    import msvcrt

    handles = [msvcrt.get_osfhandle(fd) for fd in fds]
    for handle in handles:
        os.set_handle_inheritable(handle, True)
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.lpAttributeList = {"handle_list": handles}
    return handles, {"startupinfo": startupinfo, "close_fds": True}


def fd_from_parent(value: int) -> int:
    """Inverse of popen_pass_fds, called in the child."""
    if sys.platform != "win32":
        return value
    import msvcrt

    # The handle was made inheritable only to cross into this process; don't
    # leak it into this process's own children.
    os.set_handle_inheritable(value, False)
    return msvcrt.open_osfhandle(value, 0)


def wait_readable_win32(pipe: IO[bytes], timeout: float) -> bool:
    """
    Windows replacement for selecting a pipe for reading, since select() only
    supports sockets there. Returns True once data or EOF is available, False
    on timeout.
    """
    assert sys.platform == "win32"
    import _winapi
    import msvcrt

    handle = msvcrt.get_osfhandle(pipe.fileno())
    deadline = monotonic() + timeout
    delay = 0.001
    while True:
        try:
            available, _ = _winapi.PeekNamedPipe(handle, 0)
        except OSError:
            # Broken pipe: the writer is gone, so a read returns EOF.
            return True
        if available:
            return True
        if monotonic() >= deadline:
            return False
        # Back off from 1 ms to 50 ms; a benchmark usually takes far longer.
        sleep(delay)
        delay = min(delay * 2, 0.05)


def parent_process_gone(orig_ppid: int) -> bool:
    """
    True if the process `orig_ppid` that started this one has gone away.

    POSIX reparents orphans, so getppid() changes. Windows never reparents, and
    a venv's python.exe is a launcher that starts the real interpreter as its
    child, so getppid() is not the caller's pid there; check that the original
    process is still running instead.
    """
    if sys.platform != "win32":
        return orig_ppid != os.getppid()
    import _winapi

    try:
        handle = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, orig_ppid)
    except OSError:
        return True
    try:
        return _winapi.WaitForSingleObject(handle, 0) != _winapi.WAIT_TIMEOUT
    finally:
        _winapi.CloseHandle(handle)


def in_toplevel_process() -> bool:
    global _IN_TOPLEVEL_PROCESS
    return _IN_TOPLEVEL_PROCESS


# If this process dies abnormally (e.g. segfault)
# it will not shut down the workers. Instead,
# the workers will have their parent reassigned to the
# init process. This launches a separate thread to
# watch for the worker getting reassigned,
# and cleans it up in this case.
#
# This function cannot be an inner function since otherwise mp_context="spawn" would
# not work for ProcessPoolExecutor since inner functions cannot be pickled.
def _async_compile_initializer(orig_ppid: int, close_fds: tuple[int, ...] = ()) -> None:
    # In fork mode a worker inherits the sidecar's entire fd table, including the
    # sidecar<->parent pipes. The worker must not keep those open: holding the
    # result pipe's write end would stop the parent from ever seeing EOF when the
    # sidecar dies. The caller only passes these fds under fork; under spawn the
    # worker doesn't inherit them and the same integers would name unrelated fds
    # the fresh interpreter reused, so closing them there would be silent
    # corruption (the OSError guard wouldn't catch it) rather than a no-op.
    for fd in close_fds:
        try:
            os.close(fd)
        except OSError:
            pass

    import torch._C

    def run() -> None:
        while True:
            sleep(60)
            if parent_process_gone(orig_ppid):
                # Windows has no SIGKILL; os.kill there calls TerminateProcess.
                os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))

    global _watchdog_thread, _original_parent
    _original_parent = orig_ppid
    _watchdog_thread = Thread(target=run, daemon=True)
    _watchdog_thread.start()
    # Ignore Ctrl-C (i.e. SIGINT) sent to pool workers to avoid meaningless log spam.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Install a crash handler to print out the stacktrace for SEGV
    torch._C._initCrashHandler()

    # Set a bit to distinguish async_compile subprocesses from the toplevel process.
    global _IN_TOPLEVEL_PROCESS
    _IN_TOPLEVEL_PROCESS = False

    # Claim a heartbeat slot so the sidecar watchdog can report this worker's
    # compile phase (no-op unless the sidecar allocated the shared buffer).
    watchdog.init_worker_slot()


_watchdog_thread: Thread | None = None
_original_parent: int | None = None


def has_parent_changed() -> bool:
    if _original_parent is None:
        return True
    return parent_process_gone(_original_parent)
