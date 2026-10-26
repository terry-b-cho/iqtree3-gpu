# =============================================================================
#  IQ-TREE-GPU container image.
#  PLACEHOLDER STATUS: builds the tool correctly, but is untested in CI (no GPU runner) and the
#  image is not published anywhere yet. Verify and push to a registry at standalone release.
#
#    docker build -f containers/Dockerfile -t iqtree-gpu:0.1.0-dev .
#    docker run --gpus all -v "$PWD":/data iqtree-gpu:0.1.0-dev \
#           iqtree3 -gpu --gpu-strict -s /data/aln.faa -m TESTNEW -T 20
# =============================================================================
# CUDA 12.8 + GCC 14 is the combination the campaign was measured on. nvcc rejects a host compiler
# newer than it supports, so the pairing is not arbitrary.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libeigen3-dev libboost-dev python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/iqtree-gpu
COPY src/ ./src/
COPY build_iqtree_gpu.sh ./
COPY scripts/iqtree3-gpu-verify.sh ./scripts/
COPY examples/ ./examples/

# Clones IQ-TREE 3 at the pinned commit, applies the patch, drops in the CUDA sources and builds.
# NOTE: this step needs network access at build time.
RUN bash build_iqtree_gpu.sh /opt/iqtree-gpu/build

ENV PATH="/opt/iqtree-gpu/build:${PATH}"
# Prove the kernel is present and exact, not merely that the binary starts.
# (Requires a GPU at run time; harmless to skip during a CPU-only image build.)
# RUN bash scripts/iqtree3-gpu-verify.sh /opt/iqtree-gpu/build/iqtree3
WORKDIR /data
ENTRYPOINT []
CMD ["iqtree3", "--help"]
