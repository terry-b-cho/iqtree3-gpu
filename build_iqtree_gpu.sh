#!/usr/bin/env bash
# build_iqtree_gpu.sh -- build the GPU-accelerated IQ-TREE 3 from a pristine upstream checkout by
# applying this repo's patch + GPU source files. Makes the tool buildable without any external tree.
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#
# Requirements: git, cmake >= 3.18, a CUDA toolkit (tested 12.8) + matching gcc (tested 14.2.0).
# The Eigen 3.4 and Boost headers IQ-TREE needs are used if installed and fetched otherwise (step 3).
# On an HMS-O2-like Lmod cluster:
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

echo "== 3. header-only dependencies: Eigen 3.4 and Boost =="
# IQ-TREE needs the Eigen and Boost HEADERS (no libraries). Many clusters have neither installed and
# no module for Eigen, so a first build died here with "Eigen3 library not found". If a header set is
# not already visible (EIGEN3_INCLUDE_DIR / Boost_INCLUDE_DIR / BOOST_ROOT, CPATH, a conda prefix, or
# /usr/include), the pinned release is downloaded ONCE into .iqtree3_src/_deps/ and used from there.
DEPS="$SRCDIR/_deps"; mkdir -p "$DEPS"
_have_header () {  # _have_header relative/header.hpp DIR...   -> prints the first DIR holding it
  local h="$1"; shift; local d
  for d in "$@"; do [ -n "$d" ] && [ -f "$d/$h" ] && { printf '%s' "$d"; return 0; }; done; return 1; }
_cpath_dirs () { printf '%s' "${CPATH:-}:${CPLUS_INCLUDE_PATH:-}" | tr ':' '\n' | grep . ; }
_fetch () {  # _fetch URL SHA256 OUT   (curl or wget; verifies the checksum)
  local url="$1" sum="$2" out="$3"
  if [ ! -f "$out" ]; then
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then wget -q -O "$out" "$url"
    else echo "ERROR: need curl or wget to download $url"; exit 1; fi
  fi
  echo "$sum  $out" | sha256sum -c --quiet - || { echo "ERROR: checksum mismatch for $out"; rm -f "$out"; exit 1; }
}
# Eigen
if [ -z "${EIGEN3_INCLUDE_DIR:-}" ]; then
  EIGEN3_INCLUDE_DIR=$(_have_header Eigen/Core $(_cpath_dirs) "${CONDA_PREFIX:+$CONDA_PREFIX/include/eigen3}" \
                        /usr/include/eigen3 /usr/local/include/eigen3 "$DEPS/eigen-3.4.0" || true)
fi
if [ -z "$EIGEN3_INCLUDE_DIR" ]; then
  echo "   Eigen headers not found on this machine: fetching Eigen 3.4.0 (2.6 MB) into $DEPS"
  _fetch https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
         8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72 "$DEPS/eigen-3.4.0.tar.gz"
  tar -xzf "$DEPS/eigen-3.4.0.tar.gz" -C "$DEPS" eigen-3.4.0/Eigen eigen-3.4.0/unsupported
  EIGEN3_INCLUDE_DIR="$DEPS/eigen-3.4.0"
fi
echo "   Eigen   : $EIGEN3_INCLUDE_DIR"
# Boost (IQ-TREE uses boost/bimap and boost/math; the headers of the full release are ~150 MB)
if [ -z "${Boost_INCLUDE_DIR:-}" ]; then
  Boost_INCLUDE_DIR=$(_have_header boost/version.hpp $(_cpath_dirs) "${BOOST_ROOT:+$BOOST_ROOT/include}" "${BOOST_ROOT:-}" \
                       "${CONDA_PREFIX:+$CONDA_PREFIX/include}" /usr/include /usr/local/include "$DEPS/boost_1_87_0" || true)
fi
if [ -z "$Boost_INCLUDE_DIR" ]; then
  echo "   Boost headers not found on this machine: fetching Boost 1.87.0 (155 MB, headers only are kept) into $DEPS"
  _fetch https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz \
         f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d "$DEPS/boost_1_87_0.tar.gz"
  tar -xzf "$DEPS/boost_1_87_0.tar.gz" -C "$DEPS" boost_1_87_0/boost
  rm -f "$DEPS/boost_1_87_0.tar.gz"
  Boost_INCLUDE_DIR="$DEPS/boost_1_87_0"
fi
echo "   Boost   : $Boost_INCLUDE_DIR"
export EIGEN3_INCLUDE_DIR Boost_INCLUDE_DIR

echo "== 4. configure + build (IQTREE_GPU=ON, sm_$SM) =="
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
