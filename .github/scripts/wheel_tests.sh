#!/bin/sh
# Run the binding's test suite against the library and model an installed wheel carries.
#
#     wheel_tests.sh <python that has the wheel installed>
#
# The smoke check proves the wheel imports and reads one name; this proves the binary inside it
# passes everything the binding tests assert. The test files and the Python facade come from the
# checkout, the shared library and the weights from the wheel - the facade is the same in both,
# the binary is the part that differs per platform. POSIX sh, because Alpine has no bash.
set -eu
PY="$1"
REPOSITORY="$(cd "$(dirname "$0")/../.." && pwd)"

"$PY" -m pip install --quiet pytest

# Asked from outside the checkout, so the import resolves to the installed wheel.
installed="$(cd "${RUNNER_TEMP:-/tmp}" && "$PY" -c '
import pathlib, neurelease
package = pathlib.Path(neurelease.__file__).parent
print(next(f for f in (package / "_native").iterdir() if f.suffix in (".so", ".dll", ".dylib")))
print(package / "model")')"
installed="$(printf '%s\n' "$installed" | tr -d '\r')"
NEURELEASE_LIBRARY="$(printf '%s\n' "$installed" | sed -n 1p)"
NEURELEASE_MODELS="$(printf '%s\n' "$installed" | sed -n 2p)"
# Left empty, conftest.py would quietly fall back to the build tree and test the wrong binary.
if [ ! -f "$NEURELEASE_LIBRARY" ] || [ ! -d "$NEURELEASE_MODELS" ]; then
    echo "no library and model found in the installed wheel" >&2
    exit 1
fi
export NEURELEASE_LIBRARY NEURELEASE_MODELS
echo "testing $NEURELEASE_LIBRARY with $NEURELEASE_MODELS"

cd "$REPOSITORY/bindings/python"
"$PY" -m pytest tests -q -m "not model" -p no:cacheprovider
