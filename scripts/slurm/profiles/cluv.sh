# Profile for any DRAC/PAICE cluster when the job is submitted by cluv (`cluv submit <cluster> ... -- ...`):
# CLUSTER=cluv is set by [tool.cluv.env] in pyproject.toml. Resources come from `cluv submit` flags / the cluv
# config, not from SUBMIT_ARGS. The env is the project's locked uv env (`.venv`, built by `cluv sync`).
SUBMIT_ARGS=()
profile_activate() {
    module load cuda 2>/dev/null || true          # nvcc / CUDA_HOME for vLLM JIT; no python module (the venv has its own)
    export CUDA_HOME="${EBROOTCUDA:-${CUDA_HOME:-}}"
    VENV="${HONESTY_EVALS_VENV:-$PWD/.venv}"
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "ERROR: venv not found at $VENV (run \`cluv sync\` from the Tamia worktree)" >&2
        exit 1
    fi
    source "$VENV/bin/activate"
    : "${HF_HOME:?HF_HOME must be set (cluv sets it per cluster)}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
}
