# honesty_evals: uv environment + cluv (handoff, 2026-11-12)

From now on honesty_evals runs in a **locked uv environment** instead of the hand-built `$SCRATCH/honesty_evals_venv`,
and jobs on clusters other than Tamia are submitted with **cluv**. Cluster rules, accounts, MFA, GPU request
recipes: the cluv skill (`~/.claude/skills/cluv/SKILL.md` on tamia3). All jobs on all clusters: the job tracker
(`ssh -L 8765:localhost:8765 tamia3.alliancecan.ca`, then http://localhost:8765).

## Where things are

| what | where |
|---|---|
| branch | `uv` (from `main` @ b1ed7e0, which includes 5 local commits not yet on `origin`) |
| remote | `origin` = `Tara-Research/honesty_evals` (organisation repo; cluv pushes the working branch there) |
| Tamia worktree of `uv` | `/project/aip-gidelgau/niklas/honesty_evals-uv` (run `cluv` commands here) |
| Tamia run checkout (unchanged) | `~/honesty_evals` (`main`) |
| clone + env on other clusters | `$SCRATCH/repos/honesty_evals/.venv` |
| results | `<clone>/results` -> `$SCRATCH/logs/honesty_evals` on each cluster; `cluv sync` fetches them to Tamia `/project/aip-gidelgau/niklas/cluv-results/honesty_evals/` |

## The environment (`pyproject.toml` + `uv.lock` + `.python-version` on `uv`)

Pinned to what `$SCRATCH/honesty_evals_venv` runs (2026-11-12): **190/199 packages identical**: vllm 0.25.0,
torch 2.11.0, transformers 5.14.1, peft 0.19.1, datasets 5.0.0, flashinfer 0.6.13, xgrammar 0.2.1, openai 2.48.0,
anthropic 0.117.0. Python: Compute Canada's **3.11.5** (as the venv; `.python-version`), so DRAC/PAICE only, not Mila.
- The two vLLM plugins the venv had installed editable from sibling repos are now pinned git deps (extra
  `plugins`, pulled in by the default dependency group `cluster`): `ddo-vllm-plugin` (NiklasTUM/ddo @ 9790d31,
  `vllm_plugin/`) and `rh-steer-plugin` (NiklasTUM/reward-hacking-misalignment @ b7a320c,
  `scripts/tamia/vllm_steer_plugin`). Both are inert unless `DDO_INTERVENTION` / `RH_STEER_MODE` are set.
  **Plugin changes now need: commit + push in that repo -> bump the commit here -> `uv lock` -> `uv sync`.**
  ddo's intervened MASK runs check the plugin file hash: re-run ddo's equivalence check for a new pin.
- Added as explicit deps (top-level in the venv): anthropic, openai, pycountry, cryptography, email-validator,
  fastapi-cli, uvloop, httptools, websockets.
- Differences: PyPI torch's bundled CUDA 13.0 runtime libraries instead of the system CUDA 13.3 the Compute Canada
  build used (cuda-toolkit, nvidia-cuda-runtime, nvjitlink, cuda-tile); `torchcodec` excluded (crashes vLLM init on
  DRAC); no pillow-simd.
- `uv lock` needs the private plugin repos over SSH:
  `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0='url.git@github.com:.insteadOf' GIT_CONFIG_VALUE_0='https://github.com/' uv lock`.

## How jobs run with cluv

The eval scripts are unchanged: `scripts/slurm/*.sbatch` source `scripts/slurm/profiles/${CLUSTER}.sh`; cluv sets
`CLUSTER=cluv`, and the new **`profiles/cluv.sh`** activates `.venv`, loads `cuda`, requires `HF_HOME` (cluv sets it per
cluster to the mirrored HF cache) and goes offline. The sbatch scripts take parameters as environment variables,
which cluv doesn't forward, so **`scripts/cluv/job.sh` exports leading `VAR=value` arguments**:

```bash
cd /project/aip-gidelgau/niklas/honesty_evals-uv
~/.claude/skills/cluv/scripts/cluster-check.sh                  # never name a disconnected cluster (hangs on Duo)
cluv sync                                                       # push, update clones + envs, fetch results/
cluv submit rorqual --nodes=1 --gpus-per-node=h100:4 --cpus-per-task=8 --mem=110G --time=2:00:00 -- \
  MODEL=Qwen/Qwen3.6-27B RUN_NAME=gen_qwen36_baseline scripts/slurm/generalized.sbatch
cluv sync                                                       # results/<eval>/<run>/ -> Tamia
```
(`submit.sh --cluster tamia ...` keeps working on Tamia as before.) Resource flags per cluster: GPU recipe table in
the cluv skill (Trillium-gpu: `--gpus-per-node=4` or `1` only, no `--mem`; Killarney H100 only as whole 8-GPU nodes).

## Tamia: switching

1. Merge `uv` into `main` (adds pyproject changes, uv.lock, .python-version, profiles/cluv.sh, scripts/cluv/, this doc).
2. Build from the run checkout on scratch next to the shared uv cache (hardlinks: ~10K files, not ~100K):
   ```bash
   cd ~/honesty_evals
   GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0='url.git@github.com:.insteadOf' GIT_CONFIG_VALUE_0='https://github.com/' \
   UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/honesty_evals-uv UV_CACHE_DIR=$SCRATCH/.cache/uv uv sync --locked
   ```
3. Switch: `profiles/tamia.sh` honours `HONESTY_EVALS_VENV`: `HONESTY_EVALS_VENV=$SCRATCH/venvs/honesty_evals-uv`, or
   change its default. Note `profiles/tamia.sh` loads `python/3.11`; the venv carries its own interpreter.
4. `$SCRATCH/honesty_evals_venv` is **shared**: ddo (`check_vllm_plugin.sbatch`, `mask.sh`), task-arithmetic and
   persona scripts also point at it. Delete it (~100K scratch files) only after every repo has switched and no
   queued job uses it; search all repos without truncating the output first.

## Before real runs on other clusters (open)

1. Datasets: evals read `data/` (in git) and HF datasets via `HF_HOME` (offline); datasets only in Tamia's HF cache
   (e.g. `cais/MASK`, `truthful_qa`) must be downloaded once into each cluster's `HF_HOME` (login node).
2. The judge model(s) the evals use must be mirrored (`~/.claude/skills/cluv/scripts/hf-mirror.sh`); Qwen3.6-27B,
   gemma-4-31B-it, Llama-3.3-70B, Olmo-3.1-32B, GLM-5.3-Flash are; gpt-oss-120b is not.
3. AuditBench uses `vendor/` and `$AUDITBENCH_VENV` (`$SCRATCH/auditbench_venv`, which no longer exists on Tamia):
   not covered by this env.
4. GPU smoke test on one cluster (e.g. one small `generalized.sbatch` run) before larger campaigns.
