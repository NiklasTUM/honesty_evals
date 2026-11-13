#!/bin/bash
# cluv job wrapper: `cluv submit <cluster> [sbatch flags] -- [VAR=value ...] <command | scripts/slurm/<eval>.sbatch>`
# The scripts/slurm/*.sbatch files read their parameters from environment variables (MODEL=..., RUN_NAME=...),
# which cluv does not forward from your shell; pass them as leading VAR=value arguments instead (exported here).
# CLUSTER=cluv (from the cluv config) makes those scripts source scripts/slurm/profiles/cluv.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
while [[ $# -gt 0 && "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do export "$1"; shift; done
echo "host=$(hostname) cluster=${CLUV_CLUSTER:-?} job=$SLURM_JOB_ID git=${GIT_COMMIT:-$(git rev-parse --short HEAD)} start=$(date +%T)"
mkdir -p logs
if [[ "${1:-}" == *.sbatch || "${1:-}" == *.sh ]]; then exec bash "$@"; fi
# plain command: activate the env the same way the profile does
source scripts/slurm/profiles/cluv.sh && profile_activate
exec "$@"
