#!/usr/bin/env bash
# Builds Rapidli's engine on Linux: llama.cpp (Unsloth's fork) with Rapidli's patches, its CUDA backend, the vision
# library, and rapidli-engine itself. Everything lands in engine/bin.
#
# Needs: curl, tar, cmake, a C++17 compiler, python3, the CUDA toolkit (nvcc) of your driver (12.8 or newer for an
# RTX 50 card or gcc 14).
#   engine/build-linux.sh                 download, patch and build everything
#   LLAMA_SRC=/path/to/llama.cpp ...      use a source tree you already have (a copy of the tree an Unsloth Studio
#                                         install keeps in ~/.unsloth/llama.cpp works). Untouched at the first run:
#                                         the script keeps the original files in .orig and can then be run again.
#   CUDA_ARCH="86;89" ...                 compute capabilities to build for (default: the card in this computer;
#                                         a build to share with other computers: CUDA_ARCH="75;86;89;120")
#   PORTABLE=1 ...                        processor code for any x86-64 with AVX2, not only this computer's
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMIT=c2cf5f983a01bc2ea7a5d68a7ce4fb171e6fe335          # the llama.cpp source the patches are written for
# Unsloth publishes this source as a release download only (the commit is not in the fork's git history)
TARBALL_URL="https://github.com/unslothai/llama.cpp/releases/download/b12347-mix-241a0f3/llama.cpp-source-commit-$COMMIT.tar.gz"
TARBALL_SHA256=bc280e70bba1a486b8f294ce1b0b4a255d67efbd9fb3db2e58dba920d94d5be0
SRC="${LLAMA_SRC:-$HERE/llama-src}"
PY="${PYTHON:-python3}"
JOBS="${JOBS:-$(nproc)}"

# the CUDA compiler: on the PATH, or where the toolkit usually is
command -v nvcc >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null 2>&1 || { echo "nvcc (the CUDA toolkit) was not found: install it, or put its bin folder on the PATH"; exit 1; }

if [ ! -f "$SRC/CMakeLists.txt" ]; then
  mkdir -p "$SRC"
  TGZ="$SRC.tar.gz"
  curl -fL --retry 3 -o "$TGZ" "$TARBALL_URL"
  echo "$TARBALL_SHA256  $TGZ" | sha256sum -c - || { echo "the llama.cpp source download does not match its checksum"; exit 1; }
  tar -xzf "$TGZ" -C "$SRC" --strip-components=1
  rm -f "$TGZ"
fi

# The patches apply to untouched source only, so the original files are kept; every run patches a fresh copy of them
# and copies over only the files whose text changed (nothing is compiled again without need).
if [ ! -d "$SRC/.orig" ]; then
  mkdir -p "$SRC/.orig"
  cp -r "$SRC/src" "$SRC/.orig/src"
  cp -r "$SRC/ggml/src/ggml-cuda" "$SRC/.orig/ggml-cuda"
fi
sync_changed() {       # sync_changed FROM TO: the files of FROM that differ from those in TO
  "$PY" - "$1" "$2" <<'EOF'
import filecmp, os, shutil, sys
src, dst, n = sys.argv[1], sys.argv[2], 0
for folder, _, files in os.walk(src):
    for f in files:
        a = os.path.join(folder, f)
        b = os.path.join(dst, os.path.relpath(a, src))
        if not os.path.exists(b) or not filecmp.cmp(a, b, shallow=False):
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.copyfile(a, b)
            n += 1
print(f"{n} source files updated in {dst}")
EOF
}

# 1. Rapidli's patches over src/ (they work on engine/llama/src, as on Windows) ...
rm -rf "$HERE/llama/src" && cp -r "$SRC/.orig/src" "$HERE/llama/src"
for p in "$HERE"/llama/patch_*.py; do "$PY" -X utf8 "$p"; done
sync_changed "$HERE/llama/src" "$SRC/src"
# ... and over the CUDA backend
STAGE="$SRC/.stage"
rm -rf "$STAGE" && mkdir -p "$STAGE/ggml/src" && cp -r "$SRC/.orig/ggml-cuda" "$STAGE/ggml/src/ggml-cuda"
for p in "$HERE"/cuda/patch_*.py; do LLAMA_SRC="$STAGE" "$PY" -X utf8 "$p"; done
sync_changed "$STAGE/ggml/src/ggml-cuda" "$SRC/ggml/src/ggml-cuda"

# 2. llama.cpp with CUDA and the vision library (mtmd), as shared libraries; nothing else of it is built
ARCH=()
[ -n "${CUDA_ARCH:-}" ] && ARCH=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
# PORTABLE=1: processor code for any x86-64 with AVX2 (2013 on) instead of this computer's own, for a build to share
[ "${PORTABLE:-0}" = 1 ] && ARCH+=("-DGGML_NATIVE=OFF")
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
      -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=ON -DLLAMA_OPENSSL=OFF       -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' "${ARCH[@]}"   # the libraries find each other next to themselves
cmake --build "$SRC/build" --target llama mtmd -j "$JOBS"

# 3. the engine
mkdir -p "$HERE/bin"
find "$SRC/build" -name 'lib*.so*' -exec cp -a {} "$HERE/bin/" \;
g++ -O2 -std=c++17 -pthread -w -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/vendor" -I"$SRC/tools/mtmd" \
    "$HERE/src/engine.cpp" "$SRC/vendor/cpp-httplib/httplib.cpp" \
    -L"$HERE/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN' -o "$HERE/bin/rapidli-engine"
echo "ENGINE BUILD OK: $HERE/bin/rapidli-engine"
