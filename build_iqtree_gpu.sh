#!/usr/bin/env bash
# build_iqtree_gpu.sh -- build the GPU-accelerated IQ-TREE 3 from a pristine upstream checkout by
# applying this repo's patch + GPU source files. Makes the tool buildable without any external tree.
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#
# Requirements: git, cmake >= 3.18, a CUDA toolkit (tested 12.8) + matching gcc (tested 14.2.0),
# and the Eigen 3.4 + Boost headers (IQ-TREE's usual build deps). On an HMS-O2-like Lmod cluster:
#   module load gcc/14.2.0 && module load cuda/12.8 cmake/3.31.2   # gcc FIRST
#
# Usage:  ./build_iqtree_gpu.sh [BUILD_DIR] [SM_ARCH]
#   BUILD_DIR default ./iqtree3_gpu_build ; SM_ARCH default "70;75;80;86;89" (multi-arch, runs on
#   any of them). Pin one for a smaller binary: 89 L40S/L40, 86 A6000/A40, 80 A100, 75 RTX6000/8000,
#   70 V100. Multiple archs: "80;86;89".
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${1:-$HERE/iqtree3_gpu_build}"
# Default to a MULTI-ARCH build covering every card this tool was tested on, so a binary built on
# one node runs natively on any of them. Building for a single wrong arch is not an error -- the
# driver JITs from embedded PTX -- but it is then not native code for the card and pays a JIT cost at
# start-up. Our own production build was mistakenly configured for sm_75 while every benchmark ran on
# an sm_89 L40S; multi-arch removes that whole class of mistake.
#   70=V100/V100S  75=RTX6000/8000  80=A100  86=A6000/A40/RTX3090  89=L40/L40S/RTX4090
# Pass a single value (e.g. 89) for a smaller, faster-to-compile binary.
SM="${2:-70;75;80;86;89}"
PIN=6401e06905f65a11ccd6e45f369a1987d657d600     # IQ-TREE 3 commit this patch was made against
SRCDIR="$HERE/.iqtree3_src"

# ---- 0. preflight -------------------------------------------------------------------------------
# Check the toolchain BEFORE a multi-minute clone and a CMake run. Without this, a missing nvcc or
# Eigen surfaces as a wall of CMake output several minutes in, which is a miserable first experience
# for someone who just cloned the repo.
echo "== 0. checking prerequisites =="
miss=0
need () {  # command  human-name  hint
  if command -v "$1" >/dev/null 2>&1; then
    printf "   ok      %-8s %s\n" "$2" "$($1 --version 2>&1 | head -1)"
  else
    printf "   MISSING %-8s %s\n" "$2" "$3"; miss=1
  fi
}
need git   git   "install git"
need cmake cmake "need >= 3.18"
need nvcc  nvcc  "CUDA toolkit not on PATH -- on Lmod clusters: module load cuda/12.8"
need g++   g++   "a C++ compiler matching your CUDA version is required"

# nvcc and the host compiler must be compatible; CUDA 12.x rejects very new gcc.
if command -v nvcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
  echo "   note    nvcc $(nvcc --version | grep -oP 'release \K[0-9.]+' | head -1) with g++ $(g++ -dumpversion)"
  echo "           (if the build fails on an 'unsupported GNU version' error, load an older gcc)"
fi
if [ "$miss" -ne 0 ]; then
  echo
  echo "ERROR: missing prerequisites above. On an Lmod cluster (e.g. HMS O2), load gcc FIRST:"
  echo "           module load gcc/14.2.0 cuda/12.8 cmake/3.31.2"
  exit 1
fi
# A GPU is NOT needed to build -- only to run. Say so, so a login-node build is not alarming.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "   ok      GPU      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "   note    GPU      none visible here -- that is fine, a GPU is only needed to RUN, not build"
fi

echo "== 1. clone IQ-TREE 3 @ ${PIN:0:10} (with submodules) =="
if [ ! -d "$SRCDIR/.git" ]; then
  git clone https://github.com/iqtree/iqtree3.git "$SRCDIR"
fi
cd "$SRCDIR"
git fetch --depth 1 origin "$PIN" 2>/dev/null || git fetch origin
git checkout -q "$PIN"
git submodule update --init --recursive --depth 1

echo "== 2. apply the GPU patch + drop in the new GPU source files =="
git checkout -- .                                  # reset any prior apply
git apply --whitespace=nowarn "$HERE/src/iqtree3_gpu.patch"
cp "$HERE/src/gpu_new_files/gpu_kernels.cu" "$HERE/src/gpu_new_files/gpu_kernels.h" \
   "$HERE/src/gpu_new_files/phylotreegpu.cpp" tree/

echo "== 3. configure + build (IQTREE_GPU=ON, sm_$SM) =="
# EIGEN3_INCLUDE_DIR / Boost_INCLUDE_DIR: point at your Eigen 3.4 and Boost headers if not auto-found.
cmake -S "$SRCDIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DIQTREE_GPU=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$SM" \
      ${EIGEN3_INCLUDE_DIR:+-DEIGEN3_INCLUDE_DIR=$EIGEN3_INCLUDE_DIR} \
      ${Boost_INCLUDE_DIR:+-DBoost_INCLUDE_DIR=$Boost_INCLUDE_DIR -DBoost_NO_BOOST_CMAKE=ON}
cmake --build "$BUILD_DIR" -j "$(nproc)"

echo "== DONE =="
echo "binary: $BUILD_DIR/iqtree3"
"$BUILD_DIR/iqtree3" --version 2>&1 | head -1 || true
# Print a command the reader can paste. Both paths are given relative to the repository root, and we say
# so, because the build can be launched from anywhere and a bare relative path would not resolve.
_rel () { case "$1" in "$HERE"/*) printf './%s' "${1#$HERE/}";; *) printf '%s' "$1";; esac; }
echo "verify it reproduces CPU IQ-TREE on your GPU -- from the repository root ($HERE):"
echo "    ./scripts/iqtree3-gpu-verify.sh $(_rel "$BUILD_DIR")/iqtree3"
