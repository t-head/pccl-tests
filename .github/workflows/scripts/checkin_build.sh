#!/bin/bash
set -e

# pccl-tests build wrapper for the ubuntu24_hggcrt13 GitHub CI pipeline.
# Adapted from comm-internal-testbox/.aoneci/script/checkin_build.sh to build
# pccl-tests directly (without requiring the comm-internal-testbox repository).

if [ -z "${source_home}" ]; then
    echo "Error: source_home environment variable is not set"
    exit 1
fi

cd ${source_home}/pccl-tests
bash build.sh --use_mpi
