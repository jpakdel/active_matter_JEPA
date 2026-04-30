#!/usr/bin/env bash
# Recovery script: find any run dir under runs/ that has a non-NaN final.json
# but no eval_results.json yet, and run eval on it.
#
# Background: the original tier-1 launcher had a path-translation bug in its
# inline NaN-check (bash heredoc + Python on Windows). The check spuriously
# flagged every cell as FAILED, so eval was never run, even though all the
# trained encoders were fine. This script catches up.
#
# Run from REFACTORED_CODEBASE/:
#     bash scripts/recover_evals.sh
set -u

cd "$(dirname "$0")/.."
RUNS="$PWD/runs"
LOG="$RUNS/recover_evals.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === recover_evals starting ===" | tee -a "$LOG"

count_ok=0
count_skip=0
count_fail=0

for run_dir in "$RUNS"/*_2026*/; do
    [ -d "$run_dir" ] || continue
    name=$(basename "$run_dir")

    if [ -f "$run_dir/eval_results.json" ]; then
        echo "  SKIP (has eval): $name" | tee -a "$LOG"
        count_skip=$((count_skip + 1))
        continue
    fi
    if [ ! -f "$run_dir/final.json" ]; then
        echo "  SKIP (no final.json — incomplete training): $name" | tee -a "$LOG"
        count_skip=$((count_skip + 1))
        continue
    fi

    # Check the final loss is finite. Pass path via argv (heredoc string
    # literals don't translate on Git Bash + Windows Python).
    is_finite=$(python -c '
import json, math, sys
try:
    d = json.load(open(sys.argv[1]))
    v = d.get("final_pred_mse")
    if v is None:
        print("MISSING")
    elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        print("NAN")
    else:
        print("OK")
except Exception as e:
    print(f"ERROR: {e}")
' "$run_dir/final.json")

    if [[ "$is_finite" != "OK" ]]; then
        echo "  SKIP ($is_finite): $name" | tee -a "$LOG"
        count_skip=$((count_skip + 1))
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === recovering eval: $name ===" | tee -a "$LOG"
    if (set -o pipefail
        python scripts/eval.py --run-dir "$run_dir" 2>&1 | tee "$RUNS/${name%/}_recover.eval.log"); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval OK: $name" | tee -a "$LOG"
        count_ok=$((count_ok + 1))
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] EVAL FAILED: $name" | tee -a "$LOG"
        count_fail=$((count_fail + 1))
    fi
done

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === recovery done: $count_ok evaluated, $count_skip skipped, $count_fail failed ===" | tee -a "$LOG"

# Final summary across all runs.
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
    final_path = run_dir / "final.json"
    if not final_path.exists(): continue
    try:
        final = json.loads(final_path.read_text())
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
        if isinstance(pmse, float) and (math.isnan(pmse) or math.isinf(pmse)):
            pmse_s = "NaN"
        elif pmse is None:
            pmse_s = "n/a"
        else:
            pmse_s = f"{pmse:.4f}"
        print(f"{cell:<55s} {fmt(a_l)} {fmt(z_l)} {fmt(a_k)} {fmt(z_k)} {pmse_s:>12s}")
    except Exception as e:
        print(f"{run_dir.name}: ERROR ({e})")
PYEOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] summary written to $RUNS/tier1_summary.txt" | tee -a "$LOG"
