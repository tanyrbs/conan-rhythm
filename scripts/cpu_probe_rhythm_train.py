import argparse
import math
import time
from itertools import cycle
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from tasks.Conan.Conan import ConanTask
from tasks.Conan.dataset import ConanDataset
from utils.commons.hparams import hparams, set_hparams


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pick_tracked_params(model):
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    patterns = [
        "content_embedding.weight",
        "rhythm_adapter.module.scheduler",
        "rhythm_adapter.module.projector",
        "decoder",
        "uv_predictor",
    ]
    tracked = []
    used = set()
    for pattern in patterns:
        for name, param in named:
            if pattern in name and name not in used:
                tracked.append((name, param))
                used.add(name)
                break
    if len(tracked) < 5:
        for name, param in named:
            if name in used:
                continue
            tracked.append((name, param))
            used.add(name)
            if len(tracked) >= 5:
                break
    return tracked


def _collect_grad_stats(params):
    grad_tensors = []
    with_grad = 0
    nonfinite = 0
    max_abs = 0.0
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        with_grad += 1
        if not torch.isfinite(grad).all():
            nonfinite += 1
            continue
        grad_tensors.append(grad.detach().float().reshape(-1))
        max_abs = max(max_abs, float(grad.detach().abs().max().cpu()))
    if not grad_tensors:
        return {
            "params_with_grad": with_grad,
            "nonfinite_grad_params": nonfinite,
            "grad_abs_max": max_abs,
            "grad_abs_mean": 0.0,
        }
    flat = torch.cat(grad_tensors)
    return {
        "params_with_grad": with_grad,
        "nonfinite_grad_params": nonfinite,
        "grad_abs_max": max_abs,
        "grad_abs_mean": float(flat.abs().mean().cpu()),
    }


def _global_param_norm(params):
    total = 0.0
    for param in params:
        data = param.detach().float()
        total += float(torch.sum(data * data).cpu())
    return math.sqrt(total)


def main():
    parser = argparse.ArgumentParser(description="CPU mini-train probe for Rhythm V2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--binary_data_dir", required=True)
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--exp_name", type=str, default="debug_cpu_probe")
    parser.add_argument("--max_sentences", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    if args.steps <= 0 or args.steps > 500:
        raise ValueError("--steps must be in [1, 500].")

    torch.manual_seed(1234)
    set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=(
            f"binary_data_dir='{args.binary_data_dir}',"
            f"processed_data_dir='{args.processed_data_dir}',"
            f"ds_workers=0,max_sentences={args.max_sentences},max_tokens={args.max_tokens},"
            f"num_sanity_val_steps=0,val_check_interval=999999,max_updates={args.steps},"
            "save_gt=False,style=True,rhythm_minimal_style_only=True"
        ),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false in this environment.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    task = ConanTask()
    task.build_tts_model()
    task.model.to(device)
    task.model.train()
    optimizers = task.build_optimizer(task.model)
    optimizer = optimizers[0]

    dataset = ConanDataset(args.split, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collater,
    )
    iterator = cycle(loader)

    trainable_param_count = sum(p.numel() for p in task.gen_params if p.requires_grad)
    all_trainable_param_count = sum(p.numel() for p in task.model.parameters() if p.requires_grad)
    trainable_tensor_count = sum(1 for p in task.gen_params if p.requires_grad)
    tracked_params = _pick_tracked_params(task.model)
    tracked_initial = {name: param.detach().cpu().clone() for name, param in tracked_params}
    param_norm_start = _global_param_norm(task.gen_params)

    print(f"[cpu-probe] config={args.config}")
    print(f"[cpu-probe] device={device}")
    print(f"[cpu-probe] split={args.split} dataset_len={len(dataset)} steps={args.steps}")
    print(f"[cpu-probe] trainable_gen_params={trainable_param_count:,}")
    print(f"[cpu-probe] trainable_model_params={all_trainable_param_count:,}")
    print(f"[cpu-probe] trainable_gen_tensors={trainable_tensor_count}")
    print("[cpu-probe] tracked_params:")
    for name, _ in tracked_params:
        print(f"  - {name}")

    history = []
    start_time = time.time()
    for step in range(args.steps):
        batch = next(iterator)
        batch = _move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        task.global_step = step
        total_loss, loss_output = task._training_step(batch, step, 0)
        if total_loss is None:
            raise RuntimeError("Generator training step returned None.")
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total loss at step {step}: {total_loss}")
        total_loss.backward()
        grad_stats = _collect_grad_stats(task.gen_params)
        if grad_stats["nonfinite_grad_params"] > 0:
            raise RuntimeError(f"Non-finite gradients at step {step}: {grad_stats['nonfinite_grad_params']} params")
        grad_norm_before_clip = float(
            torch.nn.utils.clip_grad_norm_(task.gen_params, hparams["clip_grad_norm"]).cpu()
        )
        optimizer.step()

        scalar_losses = {}
        for key in (
            "L_base",
            "L_rhythm_exec",
            "L_stream_state",
            "L_pitch",
            "L_exec_speech",
            "L_exec_pause",
            "L_budget",
            "L_cumplan",
        ):
            value = _tensor_to_float(loss_output.get(key))
            if value is not None:
                scalar_losses[key] = value
        scalar_losses["total_loss"] = float(total_loss.detach().cpu())
        scalar_losses["grad_norm_before_clip"] = grad_norm_before_clip
        scalar_losses.update(grad_stats)
        history.append(scalar_losses)
        detail_parts = []
        if "L_rhythm_exec" in scalar_losses and abs(scalar_losses["L_rhythm_exec"]) > 0.0:
            detail_parts.append(f"exec={scalar_losses['L_rhythm_exec']:.4f}")
        else:
            if "L_exec_speech" in scalar_losses:
                detail_parts.append(f"exec_s={scalar_losses['L_exec_speech']:.4f}")
            if "L_exec_pause" in scalar_losses:
                detail_parts.append(f"exec_p={scalar_losses['L_exec_pause']:.4f}")
        if "L_stream_state" in scalar_losses and abs(scalar_losses["L_stream_state"]) > 0.0:
            detail_parts.append(f"state={scalar_losses['L_stream_state']:.4f}")
        else:
            if "L_budget" in scalar_losses:
                detail_parts.append(f"budget={scalar_losses['L_budget']:.4f}")
            if "L_cumplan" in scalar_losses:
                detail_parts.append(f"cumplan={scalar_losses['L_cumplan']:.4f}")
        print(
            "[cpu-probe] "
            f"step={step:02d} total={scalar_losses['total_loss']:.4f} "
            f"base={scalar_losses.get('L_base', 0.0):.4f} "
            f"pitch={scalar_losses.get('L_pitch', 0.0):.4f} "
            + " ".join(detail_parts) + " "
            f"grad_norm={grad_norm_before_clip:.4f} "
            f"grad_params={grad_stats['params_with_grad']}/{trainable_tensor_count} "
            f"grad_abs_max={grad_stats['grad_abs_max']:.4e}"
        )

    elapsed = time.time() - start_time
    param_norm_end = _global_param_norm(task.gen_params)

    def summarize(key):
        values = [item[key] for item in history if key in item]
        if not values:
            return None
        return {
            "start": values[0],
            "end": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    print("[cpu-probe] summary:")
    for key in (
        "total_loss",
        "L_base",
        "L_rhythm_exec",
        "L_stream_state",
        "L_pitch",
        "L_exec_speech",
        "L_exec_pause",
        "L_budget",
        "L_cumplan",
        "grad_norm_before_clip",
    ):
        summary = summarize(key)
        if summary is None:
            continue
        print(
            f"  - {key}: start={summary['start']:.4f} end={summary['end']:.4f} "
            f"min={summary['min']:.4f} max={summary['max']:.4f} mean={summary['mean']:.4f}"
        )

    print(
        f"  - param_norm: start={param_norm_start:.4f} end={param_norm_end:.4f} "
        f"delta={param_norm_end - param_norm_start:.4f}"
    )
    print(f"  - wall_time_sec: {elapsed:.2f}")

    print("[cpu-probe] tracked_param_deltas:")
    for name, param in tracked_params:
        initial = tracked_initial[name]
        current = param.detach().cpu()
        delta = current - initial
        delta_norm = float(torch.linalg.vector_norm(delta).cpu())
        delta_max = float(delta.abs().max().cpu())
        print(f"  - {name}: delta_norm={delta_norm:.6f} delta_abs_max={delta_max:.6e}")


if __name__ == "__main__":
    main()
