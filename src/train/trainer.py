"""D-JEPA training loop.

Structure (one epoch loop, one step loop):
    for epoch in range(num_epochs):
        for batch in loader:
            ctx = batch['context']       # (B, C, T, H, W)
            tgt = batch['target']
            z_ctx = encoder(ctx)
            with torch.no_grad():
                z_tgt = encoder(tgt).detach()   # stop-grad target branch
            z_hat = predictor(z_ctx)
            loss = DJepaLoss(z_hat, z_tgt)      # pred_mse + lam * reg
            loss.total.backward()
            opt.step(); lr_sched.step(); wd_sched.step()
            checkpoint periodically

The target encoder shares weights with the context encoder (the physics-repo
convention; see project_plan §1). Target uses no_grad to stop gradient on the
target branch — this is the regularizer's job that the EMA target encoder used
to do in V-JEPA.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch

from src.models.encoder import count_parameters, num_tokens
from src.train.builders import build_from_config, build_loader
from src.train.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_old_checkpoints,
    save_checkpoint,
    write_run_config,
)
from src.train.djepa_optim import build_optimizer_and_scheds
from src.train.manifest import append_run, update_run
from src.train.step import StepMetrics, now_ms, train_one_step


def train(
    cfg: dict,
    run_dir: str | Path,
    *,
    resume: bool = True,
    max_steps: Optional[int] = None,
    max_epochs: Optional[int] = None,
    log_cb: Optional[Callable[[StepMetrics], None]] = None,
    manifest_path: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Run D-JEPA training.

    Returns the final summary dict (wall time, final metrics).

    `max_steps` and `max_epochs` override cfg; used by smoke tests to cap a run.
    `log_cb(step_metrics)` is called after every optimizer step.
    `manifest_path` + `run_id`, if given, add a row to the TSV run tracker at
    start and update it on exit (success or crash).
    """
    import json as _json
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, cfg)
    metrics_path = run_dir / "metrics.jsonl"
    # Truncate metrics log only on fresh runs.
    if not resume or not metrics_path.exists():
        metrics_path.write_text("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))

    # Build everything.
    encoder, predictor, loss_fn = build_from_config(cfg, device)
    train_loader = build_loader(cfg, split="train", shuffle=True)

    num_epochs = max_epochs if max_epochs is not None else cfg["optim"]["num_epochs"]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = cfg["optim"]["warmup_epochs"] * steps_per_epoch

    optimizer, lr_sched, wd_sched = build_optimizer_and_scheds(
        [encoder, predictor],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        lr=cfg["optim"]["lr"],
        start_lr=cfg["optim"]["start_lr"],
        final_lr=cfg["optim"]["final_lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        final_weight_decay=cfg["optim"]["final_weight_decay"],
        wd_exclude_bias_and_norm=cfg["optim"]["wd_exclude_bias_and_norm"],
    )

    use_amp = cfg["optim"]["use_amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Register in manifest.
    if manifest_path is not None and run_id is not None:
        append_run(
            manifest_path, run_id, cfg,
            total_steps=total_steps, warmup_steps=warmup_steps,
        )

    global_step = 0
    start_epoch = 0

    # Resume if possible.
    if resume:
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt is not None:
            print(f"[resume] loading {ckpt}", flush=True)
            payload = load_checkpoint(
                ckpt,
                encoder=encoder,
                predictor=predictor,
                optimizer=optimizer,
                lr_sched=lr_sched,
                wd_sched=wd_sched,
                scaler=scaler,
                map_location=str(device),
            )
            global_step = payload["global_step"]
            start_epoch = payload["epoch"]

    # Training loop.
    t_start = time.perf_counter()
    last_metrics: Optional[StepMetrics] = None
    log_every = cfg["log"].get("log_every_steps", 10)
    save_every = cfg["log"].get("save_every_steps", 500)
    keep_last_n = cfg["log"].get("keep_last_n_ckpts", 3)

    metrics_fh = open(metrics_path, "a", buffering=1)  # line-buffered

    def _log_jsonl(m: StepMetrics):
        metrics_fh.write(_json.dumps({
            "step": m.step, "epoch": m.epoch,
            "total": m.total, "pred_mse": m.pred_mse, "sigreg": m.sigreg,
            "lr": m.lr, "wd": m.wd,
            "fwd_ms": m.fwd_ms, "bwd_ms": m.bwd_ms,
            "step_ms": m.step_ms, "data_ms": m.data_ms,
        }) + "\n")

    crashed = False
    try:
        for epoch in range(start_epoch, num_epochs):
            encoder.train()
            predictor.train()
            t_data0 = now_ms()
            for batch in train_loader:
                t_data1 = now_ms()
                out, timings = train_one_step(
                    batch,
                    encoder=encoder, predictor=predictor,
                    loss_fn=loss_fn, optimizer=optimizer,
                    lr_sched=lr_sched, wd_sched=wd_sched,
                    scaler=scaler, cfg=cfg, device=device,
                )
                global_step += 1
                last_metrics = StepMetrics(
                    step=global_step, epoch=epoch,
                    pred_mse=out.pred_mse.item(),
                    sigreg=out.sigreg.item(),
                    total=out.total.item(),
                    lr=timings["lr"], wd=timings["wd"],
                    fwd_ms=timings["fwd_ms"], bwd_ms=timings["bwd_ms"],
                    step_ms=timings["step_ms"], data_ms=t_data1 - t_data0,
                )
                _log_jsonl(last_metrics)
                if log_cb is not None:
                    log_cb(last_metrics)
                elif global_step % log_every == 0 or global_step == 1:
                    print(
                        f"[step {global_step:6d} | ep {epoch}] "
                        f"total={last_metrics.total:.4f}  "
                        f"pred_mse={last_metrics.pred_mse:.4f}  "
                        f"sigreg={last_metrics.sigreg:.4f}  "
                        f"lr={last_metrics.lr:.2e}  "
                        f"fwd={last_metrics.fwd_ms:.0f}ms  "
                        f"bwd={last_metrics.bwd_ms:.0f}ms  "
                        f"data={last_metrics.data_ms:.0f}ms",
                        flush=True,
                    )

                if save_every > 0 and global_step % save_every == 0:
                    save_checkpoint(
                        run_dir / "checkpoints" / f"ckpt_{global_step:07d}.pt",
                        encoder=encoder, predictor=predictor,
                        optimizer=optimizer, lr_sched=lr_sched, wd_sched=wd_sched,
                        global_step=global_step, epoch=epoch, config=cfg, scaler=scaler,
                    )
                    prune_old_checkpoints(run_dir, keep_last_n)

                if max_steps is not None and global_step >= max_steps:
                    break
                t_data0 = now_ms()

            # End of epoch: always save.
            save_checkpoint(
                run_dir / "checkpoints" / f"ckpt_{global_step:07d}.pt",
                encoder=encoder, predictor=predictor,
                optimizer=optimizer, lr_sched=lr_sched, wd_sched=wd_sched,
                global_step=global_step, epoch=epoch + 1, config=cfg, scaler=scaler,
            )
            prune_old_checkpoints(run_dir, keep_last_n)
            if max_steps is not None and global_step >= max_steps:
                break
    except BaseException:
        crashed = True
        raise
    finally:
        metrics_fh.close()
        if manifest_path is not None and run_id is not None:
            wall_s = time.perf_counter() - t_start
            status = "crashed" if crashed else "done"
            update_run(
                manifest_path, run_id,
                status=status,
                wall_hours=f"{wall_s/3600:.3f}",
                final_pred_mse=(f"{last_metrics.pred_mse:.6f}" if last_metrics else ""),
                final_sigreg=(f"{last_metrics.sigreg:.6f}" if last_metrics else ""),
                final_total_loss=(f"{last_metrics.total:.6f}" if last_metrics else ""),
            )

    wall_s = time.perf_counter() - t_start
    summary = {
        "wall_s": wall_s,
        "global_step": global_step,
        "final_pred_mse": last_metrics.pred_mse if last_metrics else None,
        "final_sigreg": last_metrics.sigreg if last_metrics else None,
        "final_total_loss": last_metrics.total if last_metrics else None,
        "encoder_params": count_parameters(encoder),
        "predictor_params": count_parameters(predictor),
        "total_steps_planned": total_steps,
        "warmup_steps": warmup_steps,
        "num_tokens": num_tokens(
            img_size=cfg["model"]["encoder"]["img_size"],
            num_frames=cfg["model"]["encoder"]["num_frames"],
            patch_size=cfg["model"]["encoder"]["patch_size"],
            tubelet_size=cfg["model"]["encoder"]["tubelet_size"],
        ),
    }
    # Persist summary alongside run dir.
    with open(run_dir / "final.json", "w") as f:
        _json.dump(summary, f, indent=2, default=str)
    return summary
