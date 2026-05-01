"""One-shot generator for the docs under results/.

Reads every run's eval_results.json + final.json + config.json and emits:
  - results/RUN_INVENTORY.md
  - results/METRICS.md
  - results/PARETO.md
  - results/_runs.json (raw compiled inventory for any future tooling)

Run from REFACTORED_CODEBASE/:
    python scripts/_gen_results_docs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUNS = []


def parse_new(name: str):
    parts = name.rsplit("_", 2)[0]  # drop timestamp
    if parts.startswith("baseline_"):
        routing = "baseline"; parts = parts[9:]
    elif parts.startswith("exp_a_"):
        routing = "exp_a"; parts = parts[6:]
    elif parts.startswith("exp_b_"):
        routing = "exp_b"; parts = parts[6:]
    else:
        return None
    if parts.startswith("vit_"):
        backbone = "vit"; parts = parts[4:]
    elif parts.startswith("cnn_"):
        backbone = "cnn"; parts = parts[4:]
    else:
        return None
    if parts.startswith("ema_"):
        target = "ema"; parts = parts[4:]
    elif parts.startswith("shared_"):
        target = "shared"; parts = parts[7:]
    else:
        return None
    return routing, backbone, target, parts


def collect_run(run_dir: Path, source: str, manual=None):
    er = run_dir / "eval_results.json"
    if not er.exists():
        return None
    d = json.loads(er.read_text())
    final = {}
    if (run_dir / "final.json").exists():
        final = json.loads((run_dir / "final.json").read_text())
    cfg = {}
    if (run_dir / "config.json").exists():
        cfg = json.loads((run_dir / "config.json").read_text())
    if manual is not None:
        routing, backbone, target, loss = manual
    else:
        parsed = parse_new(run_dir.name)
        if not parsed:
            return None
        routing, backbone, target, loss = parsed
    lp = d["linear_probe"]
    kn = d["knn"]
    return {
        "run_id": run_dir.name,
        "routing": routing, "backbone": backbone, "target": target, "loss": loss,
        "a_lin": lp["test_mse"][0], "z_lin": lp["test_mse"][1],
        "a_knn": kn["test_mse"][0], "z_knn": kn["test_mse"][1],
        "a_val_lin": lp["val_mse"][0], "z_val_lin": lp["val_mse"][1],
        "a_val_knn": kn["val_mse"][0], "z_val_knn": kn["val_mse"][1],
        "best_alpha": lp.get("best_alpha"),
        "best_k": kn.get("best_k"),
        "best_metric": kn.get("best_metric"),
        "final_pred_mse": final.get("final_pred_mse"),
        "final_sigreg": final.get("final_sigreg"),
        "final_total_loss": final.get("final_total_loss"),
        "wall_s": final.get("wall_s"),
        "encoder_params": final.get("encoder_params"),
        "predictor_params": final.get("predictor_params"),
        "num_tokens": final.get("num_tokens"),
        "global_step": final.get("global_step"),
        "lambda": cfg.get("loss", {}).get("lambda_sigreg"),
        "vicreg_var_w": cfg.get("loss", {}).get("vicreg_var_weight"),
        "vicreg_cov_w": cfg.get("loss", {}).get("vicreg_cov_weight"),
        "ema_decay": cfg.get("train", {}).get("ema_decay"),
        "source": source,
    }


# Walk new EMA runs
for run_dir in sorted(Path("runs").glob("*_2026*")):
    if not run_dir.is_dir():
        continue
    r = collect_run(run_dir, "REFACTORED_CODEBASE/runs/")
    if r:
        RUNS.append(r)

# Walk old shared-encoder runs (parent repo)
old_meta = [
    ("baseline_v0_20260421_152635", "baseline", "vit", "shared", "sigreg"),
    ("exp_a_v0_20260421_185035", "exp_a", "vit", "shared", "sigreg"),
    ("exp_b_v0_20260421_200801", "exp_b", "vit", "shared", "sigreg"),
    ("exp_b_lam001_v0_20260422_113047", "exp_b", "vit", "shared", "sigreg_lam001"),
    ("exp_b_lam1_v0_20260422_121613", "exp_b", "vit", "shared", "sigreg_lam1"),
    ("djepa_baseline_vicreg_v0_20260424_000719", "baseline", "vit", "shared", "vicreg"),
    ("djepa_exp_a_vicreg_v0_20260424_005651", "exp_a", "vit", "shared", "vicreg"),
    ("djepa_exp_b_vicreg_v0_20260424_014519", "exp_b", "vit", "shared", "vicreg"),
    ("djepa_exp_b_vicreg_lam001_v0_20260424_023349", "exp_b", "vit", "shared", "vicreg_lam001"),
    ("djepa_exp_b_vicreg_lam1_v0_20260424_032214", "exp_b", "vit", "shared", "vicreg_lam1"),
    ("djepa_exp_b_vicreg_varw10_v0_20260424_041038", "exp_b", "vit", "shared", "vicreg_varw10"),
    ("djepa_exp_b_vicreg_varw50_v0_20260424_054642", "exp_b", "vit", "shared", "vicreg_varw50"),
    ("djepa_exp_b_vicreg_covw5_v0_20260424_045839", "exp_b", "vit", "shared", "vicreg_covw5"),
]
for name, routing, backbone, target, loss in old_meta:
    rd = Path("../runs") / name
    if not rd.exists():
        continue
    r = collect_run(rd, "../runs/", manual=(routing, backbone, target, loss))
    if r:
        RUNS.append(r)


print(f"compiled {len(RUNS)} runs")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
with open(results_dir / "_runs.json", "w") as f:
    json.dump(RUNS, f, indent=2, default=str)


# ---- formatters --------------------------------------------------------------

def fmt(v, w=8, d=4, na="n/a"):
    if v is None:
        return f"{na:>{w}s}"
    if isinstance(v, float):
        if v != v:
            return f"{'NaN':>{w}s}"
        return f"{v:{w}.{d}f}"
    return f"{str(v):>{w}s}"


def fmt_bold_if(v, is_best, w=8, d=4, na="n/a"):
    """Like fmt but wraps the formatted value in **...** when is_best is true."""
    s = fmt(v, w=w, d=d, na=na)
    if is_best:
        return f"**{s.strip()}**"
    return s


def per_col_mins(runs, keys):
    """For each metric key, return the minimum value across the given runs.

    None values are skipped. NaN is skipped (NaN != NaN). Returns a dict mapping
    key -> the min value (or None if every value was missing).
    """
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None and r[k] == r[k]]
        out[k] = min(vals) if vals else None
    return out


def top_performing_run(runs, metric_keys=("a_lin", "z_lin", "a_knn", "z_knn")):
    """Identify the run that wins the most metric columns.

    A "win" = strictly equals the per-column min. Ties on win-count are broken
    by lowest a_knn. Returns the winning run_id (string) or None if `runs` is
    empty.
    """
    if not runs:
        return None
    mins = per_col_mins(runs, metric_keys)
    wins = {r["run_id"]: 0 for r in runs}
    for k in metric_keys:
        if mins[k] is None:
            continue
        for r in runs:
            if r.get(k) is not None and r[k] == mins[k]:
                wins[r["run_id"]] += 1
    # Tie-break on a_knn ascending (lower = better).
    ranked = sorted(
        runs,
        key=lambda r: (-wins[r["run_id"]], r["a_knn"] if r.get("a_knn") is not None else float("inf")),
    )
    return ranked[0]["run_id"]


# ---- 1. RUN_INVENTORY.md -----------------------------------------------------

METRIC_KEYS = ("a_lin", "z_lin", "a_knn", "z_knn")
TOP_RUN_ID = top_performing_run(RUNS, METRIC_KEYS)
GLOBAL_MINS = per_col_mins(RUNS, METRIC_KEYS)


def bold_id(run_id):
    """Bold the run_id if it's the top performer, else return as-is."""
    return f"**{run_id}**" if run_id == TOP_RUN_ID else run_id


with open(results_dir / "RUN_INVENTORY.md", "w", encoding="utf-8") as f:
    n_new = sum(1 for r in RUNS if r["source"].startswith("REFACTORED"))
    n_old = sum(1 for r in RUNS if r["source"].startswith("../"))
    f.write("# Run Inventory\n\n")
    f.write(f"All training runs in the project: **{len(RUNS)} total** ({n_new} EMA new, {n_old} shared-encoder pre-refactor).\n\n")
    f.write("All MSE values are on z-scored targets (constant-mean baseline = 1.0; lower is better). N_train=700, N_val=96, N_test=104.\n\n")
    f.write(
        "**Bolding key.** Per-column **bold values** mark the best (lowest MSE) for that "
        f"metric within the table. The bold **run_id** marks the top-performing model "
        f"overall — defined as the run that wins the most of the four reported test "
        f"metrics (a_lin, z_lin, a_kNN, z_kNN), with ties broken by lowest a_kNN. "
        f"Currently: `{TOP_RUN_ID}`.\n\n"
    )
    f.write("---\n\n")

    f.write("## Compact table — all runs sorted by alpha kNN test MSE\n\n")
    f.write("| run_id | routing | backbone | target | loss | a_lin | z_lin | a_kNN | z_kNN | wall (s) | source |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(RUNS, key=lambda x: x["a_knn"]):
        f.write(
            f"| {bold_id(r['run_id'])} | {r['routing']} | {r['backbone']} | {r['target']} | {r['loss']} | "
            f"{fmt_bold_if(r['a_lin'], r['a_lin'] == GLOBAL_MINS['a_lin'])} | "
            f"{fmt_bold_if(r['z_lin'], r['z_lin'] == GLOBAL_MINS['z_lin'])} | "
            f"{fmt_bold_if(r['a_knn'], r['a_knn'] == GLOBAL_MINS['a_knn'])} | "
            f"{fmt_bold_if(r['z_knn'], r['z_knn'] == GLOBAL_MINS['z_knn'])} | "
            f"{fmt(r['wall_s'], w=6, d=0)} | {r['source']} |\n"
        )
    f.write("\n---\n\n")

    f.write("## Grouped by routing\n\n")
    f.write(
        "Per-column bolding here is **per routing** — each sub-table's best is bolded "
        "within its own routing slice.\n\n"
    )
    for routing in ["baseline", "exp_a", "exp_b"]:
        routing_runs = [x for x in RUNS if x["routing"] == routing]
        # Per-routing column mins. final_pred_mse intentionally excluded — it's
        # the JEPA training loss and a lower value can mean collapse, not better
        # representations (e.g. baseline_vicreg with pred_mse=0.0015 + zeta_kNN=1.98).
        # Bolding the min there would visually reward collapse; don't.
        routing_mins = per_col_mins(
            routing_runs,
            ("a_lin", "z_lin", "a_knn", "z_knn", "a_val_lin", "z_val_lin"),
        )
        f.write(f"### {routing}\n\n")
        f.write("| backbone | target | loss | a_lin | z_lin | a_kNN | z_kNN | a val_lin | z val_lin | final_pmse |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(routing_runs, key=lambda x: x["a_knn"]):
            f.write(
                f"| {r['backbone']} | {r['target']} | {r['loss']} | "
                f"{fmt_bold_if(r['a_lin'],     r['a_lin']     == routing_mins['a_lin'])} | "
                f"{fmt_bold_if(r['z_lin'],     r['z_lin']     == routing_mins['z_lin'])} | "
                f"{fmt_bold_if(r['a_knn'],     r['a_knn']     == routing_mins['a_knn'])} | "
                f"{fmt_bold_if(r['z_knn'],     r['z_knn']     == routing_mins['z_knn'])} | "
                f"{fmt_bold_if(r['a_val_lin'], r['a_val_lin'] == routing_mins['a_val_lin'])} | "
                f"{fmt_bold_if(r['z_val_lin'], r['z_val_lin'] == routing_mins['z_val_lin'])} | "
                f"{fmt(r['final_pred_mse'])} |\n"
            )
        f.write("\n")

    f.write("---\n\n")
    f.write("## Hyperparameters per run\n\n")
    f.write(
        "Held-constant across every run: `lr=3e-4`, `batch_size=2`, `num_epochs=30`, "
        "`total_steps=10500`, `warmup_steps=700`, `weight_decay=0.05->0.4 cosine`, "
        "`grad_clip=1.0`, `seed=0`. ViT runs use AMP fp16; CNN runs use fp32.\n\n"
    )
    f.write("| run_id | lambda | vicreg_var_w | vicreg_cov_w | ema_decay | use_amp |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in sorted(RUNS, key=lambda x: x["run_id"]):
        amp = "False" if r["backbone"] == "cnn" else "True"
        f.write(
            f"| {bold_id(r['run_id'])} | {fmt(r['lambda'], w=6, d=3)} | "
            f"{fmt(r['vicreg_var_w'], w=6, d=1)} | {fmt(r['vicreg_cov_w'], w=6, d=1)} | "
            f"{fmt(r['ema_decay'], w=6, d=3)} | {amp} |\n"
        )
    f.write("\n")
print("wrote results/RUN_INVENTORY.md")


# ---- 2. METRICS.md -----------------------------------------------------------

with open(results_dir / "METRICS.md", "w", encoding="utf-8") as f:
    f.write("# Training Metrics — wall time, parameter counts, final losses\n\n")
    f.write(
        "Per-run metadata from each run's `final.json`. Parameter counts in millions; "
        "`wall_s` is end-to-end training (excludes eval).\n\n"
    )
    f.write("| run_id | backbone | encoder M | predictor M | total M | tokens | wall (s) | wall (min) | final pred_mse | final reg | final total |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(RUNS, key=lambda x: (x["backbone"], x["routing"], x["loss"])):
        e = (r["encoder_params"] or 0) / 1e6
        p = (r["predictor_params"] or 0) / 1e6
        wall = r["wall_s"] or 0
        f.write(
            f"| {r['run_id']} | {r['backbone']} | {e:.2f} | {p:.2f} | {e+p:.2f} | "
            f"{r['num_tokens']} | {fmt(wall, w=6, d=0)} | {wall/60:.1f} | "
            f"{fmt(r['final_pred_mse'])} | {fmt(r['final_sigreg'])} | {fmt(r['final_total_loss'])} |\n"
        )

    f.write("\n---\n\n## Backbone size summary\n\n")
    for backbone in ["vit", "cnn"]:
        runs_b = [r for r in RUNS if r["backbone"] == backbone]
        if not runs_b:
            continue
        e = (runs_b[0]["encoder_params"] or 0) / 1e6
        p = (runs_b[0]["predictor_params"] or 0) / 1e6
        f.write(f"- **{backbone}**: encoder ~{e:.1f}M, predictor ~{p:.1f}M, total ~{e+p:.1f}M\n")

    f.write("\n## Wall-time summary\n\n")
    for backbone in ["vit", "cnn"]:
        runs_b = [r for r in RUNS if r["backbone"] == backbone and r["wall_s"]]
        if not runs_b:
            continue
        avg = sum(r["wall_s"] for r in runs_b) / len(runs_b) / 60
        f.write(f"- **{backbone}**: {len(runs_b)} runs, avg {avg:.1f} min/run\n")
print("wrote results/METRICS.md")


# ---- 3. PARETO.md ------------------------------------------------------------

OTHERS = {
    "a_lin": ["z_lin", "a_knn", "z_knn"],
    "z_lin": ["a_lin", "a_knn", "z_knn"],
    "a_knn": ["a_lin", "z_lin", "z_knn"],
    "z_knn": ["a_lin", "z_lin", "a_knn"],
}
LABELS = {"a_lin": "alpha lin", "z_lin": "zeta lin", "a_knn": "alpha kNN", "z_knn": "zeta kNN"}


def top_n(metric_key, n=5, label=""):
    sorted_rows = sorted(RUNS, key=lambda r: r[metric_key])[:n]
    lines = [
        f"### Top {n} by {label}\n",
        f"| rank | run_id | routing | backbone | target | loss | {label} | (other 3 metrics) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(sorted_rows, 1):
        others = " / ".join(f"{LABELS[k]}={fmt(r[k], w=0)}" for k in OTHERS[metric_key])
        lines.append(
            f"| {i} | {r['run_id']} | {r['routing']} | {r['backbone']} | {r['target']} | {r['loss']} | "
            f"**{fmt(r[metric_key], w=0)}** | {others} |"
        )
    return "\n".join(lines) + "\n"


with open(results_dir / "PARETO.md", "w", encoding="utf-8") as f:
    f.write("# Pareto Frontier — Best-by-Metric\n\n")
    f.write(
        "Top-5 rankings across all runs for each of the four reported probe metrics. "
        "The bold cell is the metric value being ranked; the trailing column shows "
        "that run's other three metrics for context.\n\n---\n\n"
    )
    f.write("## alpha (active dipole strength)\n\n")
    f.write(top_n("a_lin", 5, "alpha linear test MSE"))
    f.write("\n")
    f.write(top_n("a_knn", 5, "alpha kNN test MSE"))
    f.write("\n---\n\n")
    f.write("## zeta (steric alignment)\n\n")
    f.write(top_n("z_lin", 5, "zeta linear test MSE"))
    f.write("\n")
    f.write(top_n("z_knn", 5, "zeta kNN test MSE"))
    f.write("\n---\n\n")

    f.write("## Pareto-optimal on (alpha kNN, zeta kNN)\n\n")
    f.write(
        "A run is Pareto-optimal if no other run has both lower alpha kNN AND lower "
        "zeta kNN. These are the configurations that are not strictly dominated by any "
        "other in the joint (alpha, zeta) space.\n\n"
    )
    pareto = []
    for r in RUNS:
        dominated = False
        for s in RUNS:
            if s is r:
                continue
            if (s["a_knn"] <= r["a_knn"] and s["z_knn"] <= r["z_knn"]
                    and (s["a_knn"] < r["a_knn"] or s["z_knn"] < r["z_knn"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    f.write("| run_id | routing | backbone | target | loss | alpha kNN | zeta kNN |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in sorted(pareto, key=lambda x: x["a_knn"]):
        f.write(
            f"| {r['run_id']} | {r['routing']} | {r['backbone']} | {r['target']} | {r['loss']} | "
            f"**{fmt(r['a_knn'], w=0)}** | **{fmt(r['z_knn'], w=0)}** |\n"
        )
    f.write(f"\n*({len(pareto)} of {len(RUNS)} runs lie on the (alpha, zeta) kNN Pareto frontier.)*\n\n")
print("wrote results/PARETO.md")
