#!/usr/bin/env bash
# Tier 1 CNN cells (8-13): the 6 CNN cells with VICReg variants. ViT cells
# already done; SIGReg+CNN stretch tier (cells 14-16) deliberately skipped.
#
# Defensive over the original launcher:
#   - set -o pipefail (so killed-Python doesn't masquerade as success)
#   - NaN detection on every cell's final.json after train completes; if NaN
#     detected, mark FAILED in the queue log and the manifest, do not eval
#     (eval on a NaN encoder is meaningless and would litter eval_results.json
#     with garbage)
#   - Resume logic: skip cells that already have eval_results.json
#   - Wipe partial run dirs before each cell so trainer can't resume from a
#     half-state
#
# Run from REFACTORED_CODEBASE/:
#     bash scripts/run_tier1_cnn.sh
set -u

cd "$(dirname "$0")/.."
RUNS="$PWD/runs"
LOG="$RUNS/tier1_cnn.log"
mkdir -p "$RUNS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === tier-1 CNN queue starting ===" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cells: cnn × {baseline, exp_a, exp_b} × {vicreg_no_cov, vicreg}" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] use_amp=false (CNN, per cnn.yaml override)" | tee -a "$LOG"

run_cell() {
    local idx="$1" routing="$2" backbone="$3" target="$4" loss="$5"
    local name="${routing}_${backbone}_${target}_${loss}"
    local train_log="$RUNS/${name}.train.log"
    local eval_log="$RUNS/${name}.eval.log"

    echo "" | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === [$idx/6] $name ===" | tee -a "$LOG"

    # Resume: skip if already done.
    local existing
    # Match "<name>_<timestamp>" where timestamp starts with the year. This
    # is required because some loss-preset names are prefixes of others
    # (vicreg vs vicreg_no_cov): the naive glob `${name}_*` matched the
    # longer-named cell's run dir, causing a false "SKIP (already done)".
    existing="$(ls -td "$RUNS/${routing}_${backbone}_${target}_${loss}"_2026* 2>/dev/null | head -1)"
    if [ -n "$existing" ] && [ -f "$existing/eval_results.json" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP (already done): $existing" | tee -a "$LOG"
        return 0
    fi
    if [ -n "$existing" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] wiping partial run dir $existing" | tee -a "$LOG"
        rm -rf "$existing"
    fi

    # Train, capturing the python exit code (not tee's).
    (set -o pipefail
     python scripts/train.py --routing "$routing" --backbone "$backbone" \
            --target "$target" --loss "$loss" 2>&1 | tee "$train_log")
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAIN FAILED (rc=$rc): $name (continuing queue)" | tee -a "$LOG"
        return 0
    fi

    # Locate the run dir the trainer just produced.
    local run_dir
    run_dir="$(ls -td "$RUNS/${routing}_${backbone}_${target}_${loss}"_* 2>/dev/null | head -1)"
    if [ -z "$run_dir" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval skipped: could not locate run dir for $name" | tee -a "$LOG"
        return 0
    fi

    # NaN detection: if final.json has NaN losses, the encoder is junk; do
    # not waste 5 min on eval, do not pollute eval_results.json with garbage.
    # (Cell 8 of the original AMP run is the cautionary tale here.)
    #
    # Path-translation note: Git Bash uses POSIX paths like /c/Users/... that
    # Python on Windows cannot open from string literals. We pass the path
    # via argv so Git Bash's CLI translation kicks in (the same reason
    # eval.py works), instead of embedding it in a heredoc string.
    local nan_check
    nan_check=$(python -c '
import json, math, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"NO_FINAL: {e}"); sys.exit(0)
v = d.get("final_pred_mse")
if v is None:
    print("NO_PRED_MSE")
elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
    print(f"NAN: final_pred_mse={v}")
else:
    print(f"OK: final_pred_mse={v}")
' "$run_dir/final.json")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] NaN-check: $nan_check" | tee -a "$LOG"
    if [[ "$nan_check" == NAN:* ]] || [[ "$nan_check" == NO_FINAL:* ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] CELL FAILED (NaN final loss): $name; eval skipped" | tee -a "$LOG"
        return 0
    fi

    # Eval.
    if python scripts/eval.py --run-dir "$run_dir" 2>&1 | tee "$eval_log"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval OK: $name -> $run_dir" | tee -a "$LOG"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] EVAL FAILED: $name (continuing)" | tee -a "$LOG"
    fi
}

# ---- Tier 1 CNN cells (6) ---------------------------------------------------
run_cell 1 baseline cnn ema vicreg_no_cov
run_cell 2 exp_a    cnn ema vicreg_no_cov
run_cell 3 exp_b    cnn ema vicreg_no_cov
run_cell 4 baseline cnn ema vicreg
run_cell 5 exp_a    cnn ema vicreg
run_cell 6 exp_b    cnn ema vicreg

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === tier-1 CNN queue done ===" | tee -a "$LOG"

# ---- final summary ----------------------------------------------------------
python - <<'PYEOF' | tee "$RUNS/tier1_summary.txt"
import json, math
from pathlib import Path

runs_dir = Path("runs")
print(f"{'cell':<55s} {'a_lin':>8s} {'z_lin':>8s} {'a_kNN':>8s} {'z_kNN':>8s} {'final_pmse':>12s}")
print("-" * 110)
def fmt(x):
    if x is None: return f"{'n/a':>8s}"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return f"{'NaN':>8s}"
    return f"{x:>8.4f}"

for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir(): continue
    if not (run_dir / "final.json").exists(): continue
    try:
        final = json.loads((run_dir / "final.json").read_text())
        pmse = final.get("final_pred_mse")
        eval_path = run_dir / "eval_results.json"
        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            lp = ev.get("linear_probe", {}); kn = ev.get("knn", {})
            a_l, z_l = lp.get("test_mse", [None, None])
            a_k, z_k = kn.get("test_mse", [None, None])
        else:
            a_l = z_l = a_k = z_k = None
        cell = run_dir.name.rsplit("_", 2)[0]
        pmse_s = "NaN" if (isinstance(pmse, float) and (math.isnan(pmse) or math.isinf(pmse))) else f"{pmse:.4f}"
        print(f"{cell:<55s} {fmt(a_l)} {fmt(z_l)} {fmt(a_k)} {fmt(z_k)} {pmse_s:>12s}")
    except Exception as e:
        print(f"{run_dir.name}: ERROR ({e})")
PYEOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] summary written to $RUNS/tier1_summary.txt" | tee -a "$LOG"
