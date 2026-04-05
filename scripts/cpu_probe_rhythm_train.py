import argparse
import json
import math
import os
import statistics
import time
from itertools import cycle
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.commons.single_thread_env import apply_single_thread_env, maybe_limit_torch_cpu_threads


PROBE_LOSS_KEYS = (
    "L_base",
    "L_rhythm_exec",
    "L_stream_state",
    "L_pitch",
    "L_exec_speech",
    "L_exec_pause",
    "L_budget",
    "L_prefix_state",
)

PROBE_RUNTIME_KEYS = (
    "rhythm_metric_disable_acoustic_train_path",
    "rhythm_metric_module_only_objective",
    "rhythm_metric_skip_acoustic_objective",
    "rhythm_metric_pitch_supervision_disabled",
    "rhythm_metric_missing_retimed_pitch_target",
    "rhythm_metric_acoustic_target_is_retimed",
    "rhythm_metric_acoustic_target_length_delta_before_align",
    "rhythm_metric_acoustic_target_length_mismatch_abs_before_align",
    "rhythm_metric_acoustic_target_resampled_to_output",
    "rhythm_metric_acoustic_target_trimmed_to_output",
)


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


def _safe_mean(values):
    return sum(values) / len(values) if values else None


def _safe_median(values):
    return statistics.median(values) if values else None


def _safe_p95(values):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _get_process_rss_bytes():
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _maybe_compile_model(model, *, torch_compile_mode: str, torch_compile_fullgraph: bool):
    if torch_compile_mode == "none":
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this torch build.")
    kwargs = {"fullgraph": bool(torch_compile_fullgraph)}
    if torch_compile_mode != "default":
        kwargs["mode"] = torch_compile_mode
    return torch.compile(model, **kwargs)


def _validate_probe_paths(*, binary_data_dir: str, processed_data_dir: str, split: str) -> None:
    if not binary_data_dir or not os.path.isdir(binary_data_dir):
        raise FileNotFoundError(
            f"binary_data_dir does not exist: {binary_data_dir}. Run binarize/preflight first."
        )
    if not processed_data_dir or not os.path.isdir(processed_data_dir):
        raise FileNotFoundError(
            f"processed_data_dir does not exist: {processed_data_dir}. Point the probe at the processed corpus root."
        )
    required_files = (
        os.path.join(binary_data_dir, f"{split}.idx"),
        os.path.join(binary_data_dir, f"{split}.data"),
        os.path.join(binary_data_dir, f"{split}_lengths.npy"),
    )
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Probe dataset is incomplete. Missing files: "
            + ", ".join(missing)
            + ". Run preflight or re-binarize before probing."
        )


def main():
    global torch, ConanTask, ConanDataset, set_hparams, hparams, DataLoader
    parser = argparse.ArgumentParser(description="Mini-train throughput / memory probe for Rhythm V2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--binary_data_dir", required=True)
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--max_sentences", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--torch_compile_fullgraph", action="store_true")
    parser.add_argument("--profile_json", type=str, default="")
    args = parser.parse_args()

    if args.steps <= 0 or args.steps > 5000:
        raise ValueError("--steps must be in [1, 5000].")
    if args.warmup_steps is None:
        args.warmup_steps = max(0, min(2, args.steps - 1))
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("--warmup_steps must be in [0, steps - 1].")
    _validate_probe_paths(
        binary_data_dir=args.binary_data_dir,
        processed_data_dir=args.processed_data_dir,
        split=args.split,
    )

    if args.device != "cuda":
        apply_single_thread_env()

    import torch  # type: ignore[no-redef]
    from torch.utils.data import DataLoader  # type: ignore[no-redef]

    if args.device != "cuda":
        maybe_limit_torch_cpu_threads()

    from tasks.Conan.Conan import ConanTask  # type: ignore[no-redef]
    from tasks.Conan.dataset import ConanDataset  # type: ignore[no-redef]
    from utils.commons.hparams import hparams, set_hparams  # type: ignore[no-redef]

    torch.manual_seed(1234)
    hparams_override = (
        f"binary_data_dir='{args.binary_data_dir}',"
        f"processed_data_dir='{args.processed_data_dir}',"
        f"ds_workers=0,max_sentences={args.max_sentences},max_tokens={args.max_tokens},"
        f"num_sanity_val_steps=0,val_check_interval=999999,max_updates={args.steps},"
        "save_gt=False,style=True,rhythm_minimal_style_only=True"
    )
    if args.hparams:
        hparams_override = f"{hparams_override},{args.hparams}"

    set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        hparams_str=hparams_override,
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
    task.model = _maybe_compile_model(
        task.model,
        torch_compile_mode=args.torch_compile_mode,
        torch_compile_fullgraph=args.torch_compile_fullgraph,
    )
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

    print(f"[train-probe] config={args.config}")
    print(f"[train-probe] device={device}")
    print(f"[train-probe] split={args.split} dataset_len={len(dataset)} steps={args.steps} warmup_steps={args.warmup_steps}")
    print(f"[train-probe] torch_compile_mode={args.torch_compile_mode} fullgraph={bool(args.torch_compile_fullgraph)}")
    print(f"[train-probe] trainable_gen_params={trainable_param_count:,}")
    print(f"[train-probe] trainable_model_params={all_trainable_param_count:,}")
    print(f"[train-probe] trainable_gen_tensors={trainable_tensor_count}")
    print("[train-probe] tracked_params:")
    for name, _ in tracked_params:
        print(f"  - {name}")

    history = []
    start_time = time.time()
    for step in range(args.steps):
        fetch_start = time.perf_counter()
        batch = next(iterator)
        fetch_time = time.perf_counter() - fetch_start
        move_start = time.perf_counter()
        batch = _move_to_device(batch, device)
        move_time = time.perf_counter() - move_start
        optimizer.zero_grad(set_to_none=True)
        task.global_step = step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - step_start

        scalar_losses = {}
        for key in PROBE_LOSS_KEYS + PROBE_RUNTIME_KEYS:
            value = _tensor_to_float(loss_output.get(key))
            if value is not None:
                scalar_losses[key] = value
        scalar_losses["total_loss"] = float(total_loss.detach().cpu())
        scalar_losses["grad_norm_before_clip"] = grad_norm_before_clip
        scalar_losses["fetch_time_sec"] = fetch_time
        scalar_losses["move_time_sec"] = move_time
        scalar_losses["step_time_sec"] = step_time
        scalar_losses["cpu_rss_bytes"] = _get_process_rss_bytes()
        if device.type == "cuda":
            scalar_losses["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            scalar_losses["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
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
            prefix_state_value = scalar_losses.get("L_prefix_state", scalar_losses.get("L_cumplan"))
            if prefix_state_value is not None:
                detail_parts.append(f"prefix_state={prefix_state_value:.4f}")
        if scalar_losses.get("rhythm_metric_module_only_objective", 0.0) > 0.5:
            detail_parts.append("module_only=1")
        if scalar_losses.get("rhythm_metric_skip_acoustic_objective", 0.0) > 0.5:
            detail_parts.append("skip_acoustic=1")
        if scalar_losses.get("rhythm_metric_pitch_supervision_disabled", 0.0) > 0.5:
            detail_parts.append("pitch_off=1")
        if scalar_losses.get("rhythm_metric_missing_retimed_pitch_target", 0.0) > 0.5:
            detail_parts.append("retimed_pitch_missing=1")
        if scalar_losses.get("rhythm_metric_acoustic_target_resampled_to_output", 0.0) > 0.5:
            detail_parts.append("retimed_resample=1")
        if scalar_losses.get("rhythm_metric_acoustic_target_trimmed_to_output", 0.0) > 0.5:
            detail_parts.append("retimed_trim=1")
        align_mismatch = scalar_losses.get("rhythm_metric_acoustic_target_length_mismatch_abs_before_align")
        if align_mismatch is not None and align_mismatch > 0.0:
            detail_parts.append(f"align_abs={align_mismatch:.1f}")
        mem_part = ""
        if scalar_losses.get("cpu_rss_bytes") is not None:
            mem_part += f" rss={(scalar_losses['cpu_rss_bytes'] / (1024 ** 2)):.1f}MB"
        if "cuda_peak_allocated_bytes" in scalar_losses:
            mem_part += f" cuda_peak={(scalar_losses['cuda_peak_allocated_bytes'] / (1024 ** 2)):.1f}MB"
        print(
            "[train-probe] "
            f"step={step:02d} total={scalar_losses['total_loss']:.4f} "
            f"base={scalar_losses.get('L_base', 0.0):.4f} "
            f"pitch={scalar_losses.get('L_pitch', 0.0):.4f} "
            + " ".join(detail_parts) + " "
            f"fetch={fetch_time * 1000.0:.1f}ms "
            f"move={move_time * 1000.0:.1f}ms "
            f"step={step_time * 1000.0:.1f}ms "
            f"grad_norm={grad_norm_before_clip:.4f} "
            f"grad_params={grad_stats['params_with_grad']}/{trainable_tensor_count} "
            f"grad_abs_max={grad_stats['grad_abs_max']:.4e}"
            f"{mem_part}"
        )

    elapsed = time.time() - start_time
    param_norm_end = _global_param_norm(task.gen_params)
    measured_history = history[args.warmup_steps:]

    def summarize(key):
        values = [item[key] for item in measured_history if key in item and item[key] is not None]
        if not values:
            return None
        return {
            "start": values[0],
            "end": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    throughput_summary = {}
    for key in (
        "fetch_time_sec",
        "move_time_sec",
        "step_time_sec",
        "cpu_rss_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ):
        values = [item[key] for item in measured_history if key in item and item[key] is not None]
        if not values:
            continue
        throughput_summary[key] = {
            "mean": _safe_mean(values),
            "median": _safe_median(values),
            "p95": _safe_p95(values),
            "max": max(values),
            "min": min(values),
        }

    print("[train-probe] summary:")
    for key in (
        "total_loss",
        *PROBE_LOSS_KEYS,
        "grad_norm_before_clip",
    ):
        summary = summarize(key)
        if summary is None:
            continue
        print(
            f"  - {key}: start={summary['start']:.4f} end={summary['end']:.4f} "
            f"min={summary['min']:.4f} max={summary['max']:.4f} mean={summary['mean']:.4f}"
        )
    runtime_summaries = {key: summarize(key) for key in PROBE_RUNTIME_KEYS}
    runtime_summaries = {key: value for key, value in runtime_summaries.items() if value is not None}
    if runtime_summaries:
        print("[train-probe] runtime_observability:")
        for key in PROBE_RUNTIME_KEYS:
            summary = runtime_summaries.get(key)
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
    if "step_time_sec" in throughput_summary and throughput_summary["step_time_sec"]["mean"]:
        mean_step = throughput_summary["step_time_sec"]["mean"]
        print(
            f"  - throughput: mean_step_ms={mean_step * 1000.0:.2f} "
            f"median_step_ms={throughput_summary['step_time_sec']['median'] * 1000.0:.2f} "
            f"p95_step_ms={throughput_summary['step_time_sec']['p95'] * 1000.0:.2f} "
            f"steps_per_sec={1.0 / mean_step:.3f}"
        )
    if "cpu_rss_bytes" in throughput_summary:
        print(
            f"  - cpu_rss_mb: mean={throughput_summary['cpu_rss_bytes']['mean'] / (1024 ** 2):.2f} "
            f"peak={throughput_summary['cpu_rss_bytes']['max'] / (1024 ** 2):.2f}"
        )
    if "cuda_peak_allocated_bytes" in throughput_summary:
        print(
            f"  - cuda_peak_allocated_mb: mean={throughput_summary['cuda_peak_allocated_bytes']['mean'] / (1024 ** 2):.2f} "
            f"peak={throughput_summary['cuda_peak_allocated_bytes']['max'] / (1024 ** 2):.2f}"
        )
    if "cuda_peak_reserved_bytes" in throughput_summary:
        print(
            f"  - cuda_peak_reserved_mb: mean={throughput_summary['cuda_peak_reserved_bytes']['mean'] / (1024 ** 2):.2f} "
            f"peak={throughput_summary['cuda_peak_reserved_bytes']['max'] / (1024 ** 2):.2f}"
        )

    print("[train-probe] tracked_param_deltas:")
    for name, param in tracked_params:
        initial = tracked_initial[name]
        current = param.detach().cpu()
        delta = current - initial
        delta_norm = float(torch.linalg.vector_norm(delta).cpu())
        delta_max = float(delta.abs().max().cpu())
        print(f"  - {name}: delta_norm={delta_norm:.6f} delta_abs_max={delta_max:.6e}")

    if args.profile_json:
        profile_payload = {
            "config": args.config,
            "binary_data_dir": args.binary_data_dir,
            "processed_data_dir": args.processed_data_dir,
            "split": args.split,
            "device": str(device),
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "torch_compile_mode": args.torch_compile_mode,
            "torch_compile_fullgraph": bool(args.torch_compile_fullgraph),
            "trainable_gen_params": trainable_param_count,
            "trainable_model_params": all_trainable_param_count,
            "trainable_gen_tensors": trainable_tensor_count,
            "wall_time_sec": elapsed,
            "param_norm": {
                "start": param_norm_start,
                "end": param_norm_end,
                "delta": param_norm_end - param_norm_start,
            },
            "loss_summary": {
                key: summarize(key)
                for key in (
                    "total_loss",
                    *PROBE_LOSS_KEYS,
                    "grad_norm_before_clip",
                )
                if summarize(key) is not None
            },
            "runtime_summary": runtime_summaries,
            "throughput_summary": throughput_summary,
        }
        profile_path = Path(args.profile_json)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train-probe] profile_json={profile_path}")


if __name__ == "__main__":
    main()
