import os
import sys
import signal
import atexit
import pathlib
import tempfile
import traceback


class ResourceManager:
    """A utility class to manage process and file related resources.

    An instance of this class is callable and when called, it first removes all
    previously registered temporary files and then kills the registered processes.
    """

    def __init__(self):
        self.temp_files = set()
        self.processes = set()
        self._tmp_folder = pathlib.Path(os.environ.get('PYSIMT_TMP', '/tmp'))
        if not self._tmp_folder.exists():
            self._tmp_folder.mkdir(parents=True, exist_ok=True)

    def register_tmp_file(self, tmp_file: str):
        """Add new temporary file to global set."""
        self.temp_files.add(pathlib.Path(tmp_file))

    def register_proc(self, pid: int):
        """Add new process to global set."""
        self.processes.add(pid)

    def unregister_proc(self, pid: int):
        """Remove given PID from global set."""
        self.processes.remove(pid)

    def get_temp_file(self, delete: bool = False, close: bool = False):
        """Creates a new temporary file under $PYSIMT_TMP folder."""
        prefix = str(self._tmp_folder / f"pysimt_{os.getpid()}")
        t = tempfile.NamedTemporaryFile(
            mode='w', prefix=prefix, delete=delete)
        self.register_tmp_file(t.name)
        if close:
            t.close()
        return t

    def __call__(self):
        """Cleanup registered temp files and kill PIDs."""
        for tmp_file in filter(lambda x: x.exists(), self.temp_files):
            tmp_file.unlink()

        for proc in self.processes:
            try:
                os.kill(proc, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def __repr__(self):
        repr_ = "Resource Manager\n"
        if len(self.processes) > 0:
            repr_ += "Tracking Processes\n"
            for proc in self.processes:
                repr_ += " {}\n".format(proc)

        if len(self.temp_files) > 0:
            repr_ += "Tracking Temporary Files\n"
            for tmp_file in self.temp_files:
                repr_ += " {}\n".format(tmp_file)

        return repr_

    @staticmethod
    def _register_exception_handler(logger, quit_on_exception=False):
        """Setup exception handler."""

        def exception_handler(exctype, val, trace):
            """Let Python call this when an exception is uncaught."""
            logger.info(
                ''.join(traceback.format_exception(exctype, val, trace)))

        def exception_handler_quits(exctype, val, trace):
            """Let Python call this when an exception is uncaught."""
            logger.info(
                ''.join(traceback.format_exception(exctype, val, trace)))
            sys.exit(1)

        if quit_on_exception:
            sys.excepthook = exception_handler_quits
        else:
            sys.excepthook = exception_handler

    @staticmethod
    def register_handler(logger):
        """Register atexit and signal handlers."""
        # Register exit handler
        atexit.register(res_mgr)

        # Register SIGINT and SIGTERM
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        ResourceManager._register_exception_handler(logger)


# Create a global cleaner
res_mgr = ResourceManager()


def _signal_handler(signum, frame):
    """Let Python call this when SIGINT or SIGTERM caught."""
    res_mgr()
    sys.exit(0)
