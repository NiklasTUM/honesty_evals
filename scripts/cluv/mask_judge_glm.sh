#!/bin/bash
# MASK judge + metrics + process on existing generations, with GLM-5.3-Flash as the judge instead of the default
# in-job gpt-oss-120b. GLM needs vLLM >= 0.30 (this env has 0.25), so it is served from ddo's locked judge env
# (ddo envs/judge, vLLM 0.32; clone at $SCRATCH/repos/ddo) and judged through the `openai` backend against
# that local server.
#
#   cluv submit fir --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0 --time=3:00:00 -- \
#       scripts/cluv/mask_judge_glm.sh <run_dir_name> [...]
#
# Each <run_dir_name> is results/ddo_mask_glm53/<name>/ holding a MASK run's config.json and */responses/ only
# (copy them there first; never old evaluated/ files, the judge would resume from them). Results stay in that
# directory (results/ -> cluv fetches it to Tamia). The judge resumes from partial outputs and its request cache,
# so a job that times out is continued by submitting it again.
#
# Judge settings: honesty_evals' configs/judge.yaml (temperature 1.0, top_p 1.0, max_tokens 8192, seed) with
# reasoning effort "max": GLM-5.3's chat template runs anything but low/high at Max. The server's glm45
# reasoning parser keeps the reasoning out of the reply content, so the answer parsers see only the answer.
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: mask_judge_glm.sh <run_dir_name> [...]" >&2; exit 1; }
: "${HF_HOME:?HF_HOME must be set (cluv sets it per cluster)}"
GLM="$HF_HOME/hub/models--Intel--GLM-5.3-Flash-W4A16-AutoRound/snapshots/5eee1846f0321058ed73745f9aa16f2aaf0fc0a0"
JUDGE_VENV="${DDO_JUDGE_VENV:-$SCRATCH/repos/ddo/envs/judge/.venv}"
[ -f "$GLM/config.json" ] || { echo "no GLM-5.3-Flash snapshot at $GLM" >&2; exit 1; }
[ -x "$JUDGE_VENV/bin/python" ] || { echo "no judge env at $JUDGE_VENV (cluv run <cluster> -- uv sync --project envs/judge, in ddo)" >&2; exit 1; }
for name in "$@"; do
    d="results/ddo_mask_glm53/$name"
    [ -f "$d/config.json" ] || { echo "missing $d/config.json" >&2; exit 1; }
done

# Server, as ddo's slurm/llm.sbatch: the env's own CUDA 13 nvcc for FlashInfer JIT, no module CPATH, private HOME,
# no custom all-reduce, eager (CUDA-graph capture crashed vLLM judges on these nodes).
logdir="$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')")"
server_log="$logdir/vllm-$SLURM_JOB_ID.log"
nvcc=$(ls -d "$JUDGE_VENV"/lib/python3*/site-packages/nvidia/cu*/bin/nvcc 2>/dev/null | head -1 || true)
[ -n "$nvcc" ] || { echo "no CUDA nvcc in $JUDGE_VENV" >&2; exit 1; }
cuda="${nvcc%/bin/nvcc}"
mkdir -p "$SLURM_TMPDIR/home"
# FlashInfer links its JIT kernels with -lcudart -lcuda; the cuda module puts them on LIBRARY_PATH (as ddo's
# slurm/env.sh does on Tamia; without it: "ld: cannot find -lcudart", Fir job 3694607). Its CPATH is unset below.
set +u; module load gcc cuda; set -u
env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u VIRTUAL_ENV \
    PATH="$JUDGE_VENV/bin:$PATH" HOME="$SLURM_TMPDIR/home" CUDA_HOME="$cuda" \
    LD_LIBRARY_PATH="$cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/inductor" \
    VLLM_CACHE_ROOT="$SLURM_TMPDIR/vllm" HF_HUB_OFFLINE=1 \
    "$JUDGE_VENV/bin/python" -m vllm.entrypoints.cli.main serve "$GLM" --served-model-name llm \
    --tensor-parallel-size 4 --max-model-len 16384 --reasoning-parser glm45 \
    --disable-custom-all-reduce --enforce-eager --port 8000 > "$server_log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
deadline=$((SECONDS + 2400))
until curl -sf localhost:8000/v1/models > /dev/null; do
    kill -0 $server 2>/dev/null || { echo "vLLM exited during startup; see $server_log" >&2; exit 1; }
    grep -qE "Engine core initialization failed|WorkerProc initialization failed|Worker failed with error" "$server_log" \
        && { echo "vLLM engine failed during startup; see $server_log" >&2; exit 1; }
    [ $SECONDS -lt $deadline ] || { echo "vLLM not ready after 2400 s; see $server_log" >&2; exit 1; }
    sleep 15
done
echo "GLM-5.3-Flash ready ($(date +%T)); server log $server_log"

source scripts/slurm/profiles/cluv.sh && profile_activate
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
    API_JUDGE_CONCURRENCY=128 API_JUDGE_TIMEOUT=1800 API_JUDGE_CACHE="$SCRATCH/.cache/api_judge_glm53"
for name in "$@"; do
    d="results/ddo_mask_glm53/$name"
    echo "=== $name $(date +%T)"
    honesty-evals mask judge --run-dir "$d" --judge-backend openai --judge-model llm --reasoning-effort max \
        --batch-size 1000
    honesty-evals mask metrics --run-dir "$d"
    honesty-evals mask process --run-dir "$d"
done
echo "done $(date +%T)"
