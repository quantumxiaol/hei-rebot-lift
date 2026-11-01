"""Launch official mjpython with the shared-library path of uv's Python."""

import os
import platform
import runpy
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    if platform.system() != "Darwin":
        raise SystemExit("This viewer launcher is for macOS only")
    # mjpython derives @executable_path from .venv/bin/python, while uv stores
    # libpython in the base interpreter's lib directory. Set this inside Python:
    # macOS system executables such as /usr/bin/env can strip DYLD variables.
    library_dir = sysconfig.get_config_var("LIBDIR")
    if library_dir:
        fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{library_dir}:{fallback}"
    launcher = Path(sys.executable).parent / "mjpython"
    sys.argv[0] = str(launcher)
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
