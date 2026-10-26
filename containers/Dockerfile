# =============================================================================
#  IQ-TREE3-GPU container image.
#
#  Build FROM THE REPOSITORY ROOT (the COPY paths are relative to the build context; nothing is
#  fetched from GitHub, so the recipe works identically in any clone or fork):
#
#    docker build -f containers/Dockerfile -t iqtree3-gpu:0.1.0 .
#    docker run --rm --gpus all -v "$PWD":/data iqtree3-gpu:0.1.0 \
#           iqtree3 -gpu --gpu-strict -s /data/aln.faa -m TESTNEW -T 20
#
#  --gpus all needs the NVIDIA Container Toolkit on the host. Building needs network access (the
#  build script clones IQ-TREE 3 at the pinned commit) and takes 10-20 minutes. CI builds this image
#  on every push and runs the worked example inside it on the CPU; GPU verification is yours:
#
#    docker run --rm --gpus all iqtree3-gpu:0.1.0 bash /opt/iqtree3-gpu/scripts/iqtree3-gpu-verify.sh iqtree3
#    docker run --rm --gpus all -v "$PWD":/work -w /work/examples iqtree3-gpu:0.1.0 ./run_example.sh iqtree3   # from your clone
#
#  CUDA 12.8 + GCC 13 (Ubuntu 24.04's) is a combination nvcc accepts; the campaign was measured with
#  GCC 14.2.0 on the cluster, and the kernel is the same source either way.
# =============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libeigen3-dev libboost-dev python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/iqtree3-gpu
COPY build_iqtree_gpu.sh LICENSE VERSION ./
COPY src/ ./src/
COPY scripts/iqtree3-gpu-verify.sh scripts/rf_logl_check.py ./scripts/

# Clones IQ-TREE 3 at the pinned commit, applies the patch, drops in the CUDA sources and builds a
# multi-arch binary (sm_70 to sm_89). Eigen and Boost come from the packages installed above.
RUN bash build_iqtree_gpu.sh /opt/iqtree3-gpu/build \
    && rm -rf /opt/iqtree3-gpu/.iqtree3_src /opt/iqtree3-gpu/build/CMakeFiles

ENV PATH="/opt/iqtree3-gpu/build:${PATH}"
WORKDIR /data
ENTRYPOINT []
CMD ["iqtree3", "--help"]
