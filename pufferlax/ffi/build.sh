#!/bin/bash
# Build the pufferlax GPU FFI for one env (default: craftax), linking the env
# from vendored pufferlib sources (read-only). pufferlib is never modified.
set -euo pipefail

ENV_NAME="${1:-craftax}"
PUF=/home/farr/pufferlax/vendor/pufferlib
RBW_VENV=/home/farr/rl-in-big-worlds/.venv
CUDA_HOME=/usr/local/cuda
RAYLIB=raylib-5.5_linux_amd64
ARCH="${NVCC_ARCH:-sm_120}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FFI_INC=$("$RBW_VENV/bin/python" -c "import jaxlib,os;print(os.path.join(os.path.dirname(jaxlib.__file__),'include'))")
mkdir -p "$HERE/build"

echo "[1/3] env static lib ($ENV_NAME, host C via clang)"
clang -c -O2 -DNDEBUG -mavx2 -mfma \
    -I"$PUF" -I"$PUF/src" -I"$PUF/ocean/$ENV_NAME" -I"$PUF/vendor" \
    -I"$PUF/$RAYLIB/include" -I"$CUDA_HOME/include" \
    -DPLATFORM_DESKTOP -fno-semantic-interposition -fvisibility=hidden -fPIC -fopenmp \
    "$PUF/ocean/$ENV_NAME/binding.c" -o "$HERE/build/libstatic_${ENV_NAME}.o"
ar rcs "$HERE/build/libstatic_${ENV_NAME}.a" "$HERE/build/libstatic_${ENV_NAME}.o"

echo "[2/3] FFI handler (nvcc, host C++17, $ARCH)"
"$CUDA_HOME/bin/nvcc" -c -std=c++17 -arch="$ARCH" -O2 -Xcompiler -fPIC \
    -I"$FFI_INC" -I"$CUDA_HOME/include" \
    "$HERE/puffer_ffi.cu" -o "$HERE/build/puffer_ffi.o"

echo "[3/3] link puffer_ffi_${ENV_NAME}.so"
g++ -shared -fPIC -fopenmp \
    "$HERE/build/puffer_ffi.o" "$HERE/build/libstatic_${ENV_NAME}.a" \
    "$PUF/$RAYLIB/lib/libraylib.a" \
    -L"$CUDA_HOME/lib64" -lcudart -lomp5 -O2 -Bsymbolic-functions \
    -o "$HERE/puffer_ffi_${ENV_NAME}.so"

echo "OK -> $HERE/puffer_ffi_${ENV_NAME}.so"
