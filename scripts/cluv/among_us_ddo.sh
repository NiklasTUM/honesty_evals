#!/bin/bash
# Among Us with a ddo direction (or frame) ablated in the IMPOSTORS only, run stage only (no judge), in one job
# so it can race with `cluv submit first`:
#
#   cluv submit first --time=6:00:00 --job-name=<name> -- \
#       scripts/cluv/among_us_ddo.sh <plugin_hash> <run_name> <num_games> <direction|clean> [game_config]
#
# <plugin_hash>: first 16 hex of sha256(ddo_vllm_plugin.py) that passed ddo's equivalence check; the installed
#   plugin must match it. <direction>: path under $ARTIFACTS/artifacts/ddo/directions, or `clean` for a run
#   with no intervention (one server). [game_config]: default configs/evals/among_us.yaml (7 players, 2
#   impostors, the persona_playground setup).
#
# With a direction, `honesty-evals among-us run --intervention-env` starts two servers on 2 GPUs each: the
# impostors talk to the one with the ddo plugin ablating the direction (DDO_INTERVENTION=ablate), crewmates
# to an unmodified one. Sampling as the base run amongus_qwen36_base_100g_nothink: temperature 0.7, top-p 1,
# 2048 new tokens, seed 42, thinking off. Afterwards ddo's check_vllm_plugin.py checks the ablated server's
# stream stats (hooks fired, nothing left along the direction at any layer).
# Output: results/among_us/<run_name>/ (games/, winners.json, config.json) and <run_name>.plugin_stats/;
# cluv fetches results/ to Tamia.
set -euo pipefail
[ $# -ge 4 ] || { echo "usage: among_us_ddo.sh <plugin_hash> <run_name> <num_games> <direction|clean> [game_config]" >&2; exit 1; }
want_hash="$1"; name="$2"; games="$3"; rel="$4"; game_config="${5:-configs/evals/among_us.yaml}"
: "${ARTIFACTS:?ARTIFACTS must be set (cluv sets it per cluster)}"
DDO="$SCRATCH/repos/ddo"
source scripts/slurm/profiles/cluv.sh && profile_activate

plugin=$(python -c "import ddo_vllm_plugin as m; print(m.__file__)")
have_hash=$(sha256sum "$plugin" | cut -c1-16)
[ "$have_hash" = "$want_hash" ] || { echo "installed plugin $have_hash ($plugin) != checked plugin $want_hash" >&2; exit 1; }
[ -f "$game_config" ] || { echo "no such game config: $game_config" >&2; exit 1; }

stats="$PWD/results/among_us/$name.plugin_stats"
intervention=()
if [ "$rel" != clean ]; then
    direction="$ARTIFACTS/artifacts/ddo/directions/$rel"
    [ -f "$direction" ] || { echo "no such direction file: $direction" >&2; exit 1; }
    [ -x "$DDO/.venv/bin/python" ] || { echo "no ddo clone/env at $DDO (cluv sync, in ddo)" >&2; exit 1; }
    intervention=(--intervention-env "DDO_INTERVENTION=ablate DDO_DIRECTION=$direction DDO_STATS=$stats"
                  --intervention-roles Impostor)
fi
echo "=== among us $name: $games games, impostors: $rel, config $game_config ($(date +%T))"
honesty-evals among-us run --model Qwen/Qwen3.6-27B --run-name "$name" --num-games "$games" \
    --game-config "$game_config" "${intervention[@]}" \
    --launch-server --tp 4 --port 8192 \
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --temperature 0.7 --top-p 1.0 --seed 42 --enable-thinking false
if [ "$rel" != clean ]; then
    (cd "$DDO" && PYTHONPATH=. .venv/bin/python scripts/check_vllm_plugin.py stats --out "$stats" --condition ablate)
fi
echo "done $(date +%T)"
