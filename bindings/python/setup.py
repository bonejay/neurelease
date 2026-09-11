"""Bundle the built library and the model files into the wheel.

`pip install ./bindings/python` used to install a package that was `__init__.py` and nothing else.
The loader then looked for the shared library beside itself, in site-packages, and found nothing,
so the documented `Parser()` failed everywhere but under the test fixture, which points at the
build directory by environment variable. The C++ suite and CI never install the package, so nothing
caught it until someone followed the README in a clean clone.

This copies the library CMake built and the repository's `model/` directory into the package at
build time, into the two places the loader already probes: `neurelease/_native/` and
`neurelease/model/`. The result is a platform wheel that works from any directory. The library is
self-contained (on Windows it imports only KERNEL32 and msvcrt; PCRE2 is linked statically), so
nothing else needs to travel with it.

Point at a differently placed library or model directory with NEURELEASE_LIBRARY and
NEURELEASE_MODELS, the same variables the loader honours at run time.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
LIBRARY_NAMES = ("libneurelease.dll", "neurelease.dll", "libneurelease.so", "libneurelease.dylib")
MODEL_FILES = ("segmenter.bin", "character_map.bin", "chinese_japanese.bin",
               "transliteration_japanese.bin", "transliteration_chinese.bin")


def find_library() -> Path:
    if configured := os.environ.get("NEURELEASE_LIBRARY"):
        library = Path(configured)
        if library.is_file():
            return library
        raise SystemExit(f"neurelease: NEURELEASE_LIBRARY={configured} is not a file")
    for build in ("release", "clean", "debug", "coverage"):
        for name in LIBRARY_NAMES:
            candidate = REPOSITORY / "build" / build / name
            if candidate.is_file():
                return candidate
    raise SystemExit(
        "neurelease: no built library found. Build it first -\n"
        "    cmake --preset release && cmake --build --preset release --parallel\n"
        "or set NEURELEASE_LIBRARY to the shared library to bundle.")


def find_models() -> Path:
    configured = os.environ.get("NEURELEASE_MODELS")
    directory = Path(configured) if configured else REPOSITORY / "model"
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise SystemExit(f"neurelease: model directory {directory} is missing {', '.join(missing)}; "
                         "set NEURELEASE_MODELS to a complete one.")
    return directory


class build_py(_build_py):
    """The ordinary build, then the library and the models copied in beside the package."""

    def run(self) -> None:
        super().run()
        package = Path(self.build_lib) / "neurelease"
        native = package / "_native"
        native.mkdir(parents=True, exist_ok=True)
        library = find_library()
        shutil.copy2(library, native / library.name)
        models = package / "model"
        if models.exists():
            shutil.rmtree(models)
        shutil.copytree(find_models(), models)


class BinaryDistribution(Distribution):
    """A wheel carrying a shared library is specific to a platform, and must say so in its tag."""

    def has_ext_modules(self) -> bool:  # noqa: D401 - setuptools hook
        return True


setup(cmdclass={"build_py": build_py}, distclass=BinaryDistribution)
