#!/bin/bash
# MASK on Qwen3.6-27B with a ddo direction (or frame) ablated by the ddo vLLM plugin, then GLM-5.3-Flash as the
# judge, in ONE job, so a run can race with `cluv submit first`. Counterpart of ddo's slurm/mask.sh (Tamia-only
# generate) + scripts/cluv/mask_judge_glm.sh.
#
#   cluv submit first --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0 --time=6:00:00 --job-name=<name> -- \
#       scripts/cluv/mask_ddo_glm.sh <plugin_hash> <name>=<direction> [...]
#
# <plugin_hash>: first 16 hex of sha256(ddo_vllm_plugin.py) that passed ddo's equivalence check
#   (marker $SCRATCH/ddo-runs/check_vllm_plugin/passed/<hash> on Tamia); the installed plugin must match it.
# <name>: MASK run name, e.g. ddo_cone_ABCall_N2_s0_ablate. <direction>: path under
#   $ARTIFACTS/artifacts/ddo/directions (a [5120] vector or a [5120, N] frame; a frame ablates its whole span).
#
# Generation as slurm/mask.sh: thinking off, honesty_evals' sampling (temperature 1, seed 42), each run elicits its
# own beliefs, GEN_ENV reaches the generate stage only. The plugin's stream stats are checked with ddo's
# check_vllm_plugin.py (hooks fired, accumulated stream orthogonal to the span at every layer) before judging.
# Output: results/ddo_mask_glm53/<name>/ (judged run) and results/ddo_mask_glm53/<name>.plugin_stats/; cluv fetches
# results/ to Tamia. A run whose responses already exist skips generation (resubmit to resume the judge).
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: mask_ddo_glm.sh <plugin_hash> <name>=<direction> [...]" >&2; exit 1; }
want_hash="$1"; shift
: "${ARTIFACTS:?ARTIFACTS must be set (cluv sets it per cluster)}"
DDO="$SCRATCH/repos/ddo"
[ -x "$DDO/.venv/bin/python" ] || { echo "no ddo clone/env at $DDO (cluv sync, in ddo)" >&2; exit 1; }
ARCHETYPES=continuations,disinformation,doubling_down_known_facts,known_facts,provided_facts,statistics
GEN_ROOT="$PWD/results/ddo_mask_gen"

plugin=$(.venv/bin/python -c "import ddo_vllm_plugin as m; print(m.__file__)")
have_hash=$(sha256sum "$plugin" | cut -c1-16)
[ "$have_hash" = "$want_hash" ] || { echo "installed plugin $have_hash ($plugin) != checked plugin $want_hash" >&2; exit 1; }

names=()
for arg in "$@"; do
    name="${arg%%=*}"; rel="${arg#*=}"
    [ "$name" != "$arg" ] && [ -n "$rel" ] || { echo "expected <name>=<direction>, got '$arg'" >&2; exit 1; }
    direction="$ARTIFACTS/artifacts/ddo/directions/$rel"
    [ -f "$direction" ] || { echo "no such direction file: $direction" >&2; exit 1; }
    out="results/ddo_mask_glm53/$name"
    stats="$PWD/results/ddo_mask_glm53/$name.plugin_stats"
    if [ -f "$out/config.json" ]; then
        echo "=== $name: responses exist in $out, generation skipped"
    else
        echo "=== $name: generate with $direction ablated ($(date +%T))"
        [ ! -e "$GEN_ROOT/mask/$name" ] || { echo "$GEN_ROOT/mask/$name exists from an earlier failed run; remove it" >&2; exit 1; }
        STAGES=generate MODEL=Qwen/Qwen3.6-27B RUN_NAME="$name" ARCHETYPES="$ARCHETYPES" ENABLE_THINKING=false \
            HONESTY_EVALS_RESULTS="$GEN_ROOT" \
            GEN_ENV="DDO_INTERVENTION=ablate DDO_DIRECTION=$direction DDO_STATS=$stats" \
            bash scripts/slurm/mask.sbatch
        mkdir -p results/ddo_mask_glm53
        mv "$GEN_ROOT/mask/$name" "$out"
    fi
    (cd "$DDO" && PYTHONPATH=. .venv/bin/python scripts/check_vllm_plugin.py stats --out "$stats" --condition ablate)
    names+=("$name")
done
bash scripts/cluv/mask_judge_glm.sh "${names[@]}"
