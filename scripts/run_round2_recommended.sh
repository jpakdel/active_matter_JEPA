#!/usr/bin/env bash
# Round-2 sweep: 8 cells recommended by the surrogate model after refresh on
# 26 observed cells (Spearman ρ_alpha=0.60, ρ_zeta=0.19).
#
# Cells were the dedupe of the curated alpha-top5 + zeta-top5 in
# predictions/SUMMARY.md (curated = hamming==1, no Mode 3, no Mode 1 at shared).
#
# Defensive same as run_tier1_cnn.sh:
#   - set -o pipefail (killed Python doesn't masquerade as success)
#   - NaN detection on final.json before eval
#   - Resume logic: skip cells with eval_results.json already
#   - Wipe partial run dirs before each cell
#
# Run from REFACTORED_CODEBASE/:
#     bash scripts/run_round2_recommended.sh
set -u

cd "$(dirname "$0")/.."
RUNS="$PWD/runs"
LOG="$RUNS/round2_recommended.log"
mkdir -p "$RUNS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === round-2 surrogate-recommended queue starting ===" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cells: 8 (curated alpha+zeta top-5, deduped)" | tee -a "$LOG"

run_cell() {
    local idx="$1" routing="$2" backbone="$3" target="$4" loss="$5" reason="$6"
    local name="${routing}_${backbone}_${target}_${loss}"
    local train_log="$RUNS/${name}.train.log"
    local eval_log="$RUNS/${name}.eval.log"

    echo "" | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === [$idx/8] $name  ($reason) ===" | tee -a "$LOG"

    # Resume: skip if already done. Match name + 2026-timestamp suffix so
    # loss-name-prefix collisions (vicreg vs vicreg_no_cov) don't cause false
    # SKIPs.
    local existing
    existing="$(ls -td "$RUNS/${name}"_2026* 2>/dev/null | head -1)"
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
    run_dir="$(ls -td "$RUNS/${name}"_* 2>/dev/null | head -1)"
    if [ -z "$run_dir" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval skipped: could not locate run dir for $name" | tee -a "$LOG"
        return 0
    fi

    # NaN detection: eval on a NaN encoder is meaningless and pollutes the
    # eval_results.json stream with garbage. Path passed via argv so Git Bash
    # CLI translation rewrites POSIX -> Windows for Python (same trick as
    # run_tier1_cnn.sh).
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

# ---- 8 cells, ordered: alpha-strongest first, then zeta-strongest ----------
# Reason codes:
#   a#  = rank in curated alpha top-5
#   z#  = rank in curated zeta top-5
run_cell 1 baseline vit ema    vicreg_no_cov   "a1 (pred 0.136)"
run_cell 2 baseline vit ema    vicreg_lam001   "a3+z1 (pred a=0.207, z=0.385)"
run_cell 3 baseline vit ema    sigreg_lam1     "a2+z5 (pred a=0.203, z=0.442)"
run_cell 4 baseline vit ema    sigreg_lam001   "a4 (pred 0.210)"
run_cell 5 baseline vit ema    vicreg_varw10   "a5 (pred 0.210)"
run_cell 6 exp_a    vit ema    vicreg_lam001   "z3 (pred 0.411)"
run_cell 7 exp_b    cnn ema    vicreg_lam001   "z2 (pred 0.410)"
run_cell 8 exp_a    vit shared vicreg_lam001   "z4 (pred 0.426)"

echo "" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === round-2 queue done ===" | tee -a "$LOG"

# ---- final summary ----------------------------------------------------------
python - <<'PYEOF' | tee "$RUNS/round2_summary.txt"
import json, math
from pathlib import Path

CELLS = [
    "baseline_vit_ema_vicreg_no_cov",
    "baseline_vit_ema_vicreg_lam001",
    "baseline_vit_ema_sigreg_lam1",
    "baseline_vit_ema_sigreg_lam001",
    "baseline_vit_ema_vicreg_varw10",
    "exp_a_vit_ema_vicreg_lam001",
    "exp_b_cnn_ema_vicreg_lam001",
    "exp_a_vit_shared_vicreg_lam001",
]

runs_dir = Path("runs")
print(f"{'cell':<40s} {'a_lin':>8s} {'z_lin':>8s} {'a_kNN':>8s} {'z_kNN':>8s} {'final_pmse':>12s}")
print("-" * 92)

def fmt(x):
    if x is None: return f"{'n/a':>8s}"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return f"{'NaN':>8s}"
    return f"{x:>8.4f}"

for cell in CELLS:
    matches = sorted(runs_dir.glob(f"{cell}_2026*"))
    if not matches:
        print(f"{cell:<40s} (no run dir)")
        continue
    run_dir = matches[-1]  # newest
    final_path = run_dir / "final.json"
    eval_path = run_dir / "eval_results.json"
    pmse = None
    if final_path.exists():
        try:
            pmse = json.loads(final_path.read_text()).get("final_pred_mse")
        except Exception:
            pass
    a_l = z_l = a_k = z_k = None
    if eval_path.exists():
        try:
            ev = json.loads(eval_path.read_text())
            lp = ev.get("linear_probe", {}); kn = ev.get("knn", {})
            a_l, z_l = lp.get("test_mse", [None, None])
            a_k, z_k = kn.get("test_mse", [None, None])
        except Exception:
            pass
    pmse_s = "n/a" if pmse is None else (
        "NaN" if (isinstance(pmse, float) and (math.isnan(pmse) or math.isinf(pmse))) else f"{pmse:.4f}"
    )
    print(f"{cell:<40s} {fmt(a_l)} {fmt(z_l)} {fmt(a_k)} {fmt(z_k)} {pmse_s:>12s}")
PYEOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] summary written to $RUNS/round2_summary.txt" | tee -a "$LOG"
