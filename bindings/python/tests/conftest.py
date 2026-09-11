"""Make `pytest` work from anywhere in the repository without a shell preamble.

The package is not installed in a checkout, and the shared library sits wherever CMake put it, so
without this both are invisible: `python -m pytest bindings/python/tests` used to fail on the
import, or later on loading the DLL, until the caller exported PYTHONPATH and NEURELEASE_LIBRARY
by hand. An installed wheel or an explicit environment variable still wins, so CI is unaffected.
"""

import os
import sys
from pathlib import Path

BINDINGS = Path(__file__).resolve().parents[1]
REPOSITORY = BINDINGS.parents[1]

if str(BINDINGS) not in sys.path:
    sys.path.insert(0, str(BINDINGS))

if not os.environ.get("NEURELEASE_LIBRARY"):
    for build in ("release", "clean", "debug", "coverage"):
        for name in ("libneurelease.dll", "neurelease.dll", "libneurelease.so",
                     "libneurelease.dylib"):
            candidate = REPOSITORY / "build" / build / name
            if candidate.is_file():
                os.environ["NEURELEASE_LIBRARY"] = str(candidate)
                break
        if os.environ.get("NEURELEASE_LIBRARY"):
            break

if not os.environ.get("NEURELEASE_MODELS") and (REPOSITORY / "model/segmenter.bin").is_file():
    os.environ["NEURELEASE_MODELS"] = str(REPOSITORY / "model")
