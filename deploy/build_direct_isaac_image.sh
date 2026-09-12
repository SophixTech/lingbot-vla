#!/usr/bin/env bash
# Build a minimal Isaac Sim image from the already-cached official 5.1 base.
# It deliberately avoids Genie Sim's optional IK wheel, which currently fails
# its CRC check only after Docker BuildKit copies it into the build container.
set -euo pipefail

readonly dockerfile="/home/bjtc/Sophix/third_party/genie_sim/docker/Dockerfile.direct-isaac"
readonly context_dir="/tmp/sophix-isaac-direct-empty-context"
mkdir -p "$context_dir"

exec sg docker -c "docker build --progress=plain --build-arg ISAACSIM_REGISTRY=nvcr.m.daocloud.io/nvidia --build-arg ISAACSIM_TAG=5.1.0 -f '$dockerfile' -t sophix/isaac-sim:5.1 '$context_dir'"
