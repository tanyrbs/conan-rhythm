import hashlib
import os
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.Conan.Conan import Conan
from modules.Conan.rhythm_v3.g_stats import normalize_global_rate_variant
from modules.Conan.rhythm_v3.contracts import export_duration_v3_source_cache
from modules.Conan.rhythm_v3.source_cache import (
    DURATION_V3_CACHE_META_KEY,
    UNIT_LOG_PRIOR_META_KEY,
    assert_duration_v3_cache_meta_compatible,
    build_duration_v3_cache_meta,
    build_duration_v3_frontend_signature,
    build_source_rhythm_cache_v3 as build_source_rhythm_cache,
)
from modules.Emformer.emformer import EmformerDistillModel
from tasks.Conan.rhythm.streaming_commit import extract_incremental_committed_mel
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams
from utils.plot.rhythm_v3_viz import build_debug_record

try:
    from inference.streaming_runtime import (
        PrefixCodeBuffer,
        RollingMelContextBuffer,
        build_streaming_latency_report,
        build_streaming_layout_report,
        resolve_chunk_frames,
        resolve_mel_frame_ms,
        resolve_vocoder_left_context_frames,
    )
except ImportError:  # pragma: no cover - supports direct `python inference/Conan.py`
    from streaming_runtime import (
        PrefixCodeBuffer,
        RollingMelContextBuffer,
        build_streaming_latency_report,
        build_streaming_layout_report,
        resolve_chunk_frames,
        resolve_mel_frame_ms,
        resolve_vocoder_left_context_frames,
    )

__all__ = ["StreamingVoiceConversion"]


class StreamingVoiceConversion:
    """
    Streaming style-transfer inference using Emformer for feature extraction.
    """

    tokens_per_chunk: int = 4  # 4 HuBERT tokens ~= 80 ms with the default hop.

    def __init__(self, hp: Dict):
        self.hparams = hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        self.emformer = self._build_emformer()
        self.vocoder_native_streaming_capable = bool(self.vocoder.supports_native_streaming())
        self.vocoder_left_context_frames, self.vocoder_left_context_source = resolve_vocoder_left_context_frames(
            self.hparams
        )
        self.streaming_layout = build_streaming_layout_report(
            self.hparams,
            vocoder_left_context_frames=self.vocoder_left_context_frames,
            vocoder_native_streaming=self.vocoder_native_streaming_capable,
        )
        self._prompt_conditioning_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self._prompt_conditioning_cache_size = max(
            0,
            int(self.hparams.get("rhythm_prompt_conditioning_cache_size", 4) or 0),
        )
        self.rhythm_v3_cache_meta = build_duration_v3_cache_meta(
            silent_token=int(self.hparams.get("silent_token", 57)),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            emit_silence_runs=bool(self.hparams.get("rhythm_v3_emit_silence_runs", True)),
            debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
            phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        )
        self.rhythm_v3_frontend_signature = build_duration_v3_frontend_signature(
            self.rhythm_v3_cache_meta,
            g_variant=normalize_global_rate_variant(self.hparams.get("rhythm_v3_g_variant", "raw_median")),
            drop_edge_runs_for_g=int(self.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0),
            unit_prior_path=self.hparams.get("rhythm_v3_unit_prior_path"),
            summary_pool_speech_only=bool(self.hparams.get("rhythm_v3_summary_pool_speech_only", True)),
            emit_silence_runs=bool(self.hparams.get("rhythm_v3_emit_silence_runs", True)),
        )
        self.runtime_metadata = self.get_streaming_runtime_metadata()
        self.last_inference_metadata = dict(self.runtime_metadata)
        self.last_rhythm_debug_bundle = None
        self._vocoder_warm_zero()

    def _build_model(self):
        model = Conan(0, self.hparams)
        model.eval()
        load_ckpt(model, self.hparams["work_dir"], strict=False)
        return model.to(self.device)

    def _build_vocoder(self):
        vocoder_cls = get_vocoder_cls(self.hparams["vocoder"])
        if vocoder_cls is None:
            raise ValueError(
                f"Vocoder '{self.hparams['vocoder']}' is not registered. "
                "Check vocoder name and registration."
            )
        return vocoder_cls()

    def _build_emformer(self):
        emformer = EmformerDistillModel(self.hparams, output_dim=100)
        load_ckpt(emformer, self.hparams["emformer_ckpt"], strict=False)
        emformer.eval()
        return emformer.to(self.device)

    def _vocoder_warm_zero(self):
        warm_frames = max(4, int(self.vocoder_left_context_frames) + 1)
        _ = self.vocoder.spec2wav(
            np.zeros((warm_frames, self.hparams["audio_num_mel_bins"]), dtype=np.float32)
        )

    def _resolve_content_window_left_tokens(self) -> int:
        left_tokens = self.hparams.get("rhythm_infer_content_left_context_tokens")
        if left_tokens is None:
            left_tokens = self.hparams.get("rhythm_v3_infer_content_left_context_tokens")
        if left_tokens is None:
            left_tokens = max(64, int(self.hparams.get("right_context", 0)) * 8)
        return max(0, int(left_tokens))

    @staticmethod
    def _compute_content_window_start(
        *,
        total_tokens: int,
        committed_frontier_tokens: int,
        left_context_tokens: int,
    ) -> int:
        if total_tokens <= 0 or committed_frontier_tokens <= 0:
            return 0
        keep_from = max(0, int(committed_frontier_tokens) - int(left_context_tokens))
        return min(keep_from, max(0, int(total_tokens) - 1))

    @staticmethod
    def _resolve_committed_token_frontier_from_cache(
        *,
        commit_frontier: torch.Tensor | None,
        rhythm_source_cache: Dict[str, torch.Tensor] | None,
        batch_index: int = 0,
    ) -> int | None:
        if not isinstance(commit_frontier, torch.Tensor):
            return None
        if not isinstance(rhythm_source_cache, dict):
            return None
        source_duration_obs = rhythm_source_cache.get("source_duration_obs")
        if not isinstance(source_duration_obs, torch.Tensor):
            return None
        if source_duration_obs.dim() != 2 or source_duration_obs.size(0) <= batch_index:
            return None

        frontier_units = int(commit_frontier[batch_index].item())
        frontier_units = max(0, min(frontier_units, int(source_duration_obs.size(1))))
        if frontier_units <= 0:
            return 0
        committed_src = source_duration_obs[batch_index, :frontier_units].float().clamp_min(0.0)
        return int(torch.round(committed_src).sum().item())

    @staticmethod
    def _assert_monotone_committed_frontier(
        *,
        previous_frontier: int | None,
        current_frontier: int | None,
    ) -> None:
        if previous_frontier is None or current_frontier is None:
            return
        if int(current_frontier) < int(previous_frontier):
            raise RuntimeError(
                f"commit_frontier must be monotone non-decreasing: prev={int(previous_frontier)}, current={int(current_frontier)}"
            )

    @staticmethod
    def _assert_committed_prefix_not_rewritten(
        *,
        prev_state,
        next_state,
    ) -> None:
        prev_cached = getattr(prev_state, "cached_duration_exec", None) if prev_state is not None else None
        next_cached = getattr(next_state, "cached_duration_exec", None) if next_state is not None else None
        prev_committed = getattr(prev_state, "committed_units", None) if prev_state is not None else None
        next_committed = getattr(next_state, "committed_units", None) if next_state is not None else None
        if not all(isinstance(value, torch.Tensor) for value in (prev_cached, next_cached, prev_committed, next_committed)):
            return
        batch_size = min(int(prev_cached.size(0)), int(next_cached.size(0)), int(prev_committed.size(0)), int(next_committed.size(0)))
        for batch_idx in range(batch_size):
            prefix = min(
                int(prev_committed[batch_idx].item()),
                int(next_committed[batch_idx].item()),
                int(prev_cached.size(1)),
                int(next_cached.size(1)),
            )
            if prefix <= 0:
                continue
            prev_prefix = prev_cached[batch_idx, :prefix].float()
            next_prefix = next_cached[batch_idx, :prefix].float()
            if not torch.allclose(prev_prefix, next_prefix, atol=1.0e-5, rtol=1.0e-4):
                raise RuntimeError(
                    f"Committed prefix was rewritten for batch={batch_idx}, prefix_units={prefix}."
                )

    @staticmethod
    def _wav_to_mel(path: str) -> np.ndarray:
        mel = librosa_wav2spec(
            path,
            fft_size=hparams["fft_size"],
            hop_size=hparams["hop_size"],
            win_length=hparams["win_size"],
            num_mels=hparams["audio_num_mel_bins"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            sample_rate=hparams["audio_sample_rate"],
            loud_norm=hparams["loud_norm"],
        )["mel"]
        return np.clip(mel, hparams["mel_vmin"], hparams["mel_vmax"])

    def get_streaming_runtime_metadata(self, *, duration_seconds: float | None = None) -> dict[str, Any]:
        latency_report = build_streaming_latency_report(
            self.hparams,
            duration_seconds=duration_seconds,
            vocoder=self.vocoder,
        )
        return {
            "streaming_capabilities": {
                "frontend_stateful_streaming": True,
                "acoustic_decoder_native_incremental": False,
                "vocoder_native_streaming": bool(self.vocoder_native_streaming_capable),
                "full_end_to_end_stateful_streaming": False,
            },
            "vocoder_left_context_frames": int(self.vocoder_left_context_frames),
            "vocoder_left_context_source": str(self.vocoder_left_context_source),
            "streaming_layout": self.streaming_layout.to_metadata(),
            "latency_report": latency_report,
        }

    def _prepare_spk_embed(self, ref_mel_batch: torch.Tensor, spk_embed=None) -> torch.Tensor:
        if spk_embed is None:
            with torch.no_grad():
                return self.model.encode_spk_embed(ref_mel_batch.transpose(1, 2)).transpose(1, 2)
        if isinstance(spk_embed, np.ndarray):
            spk_embed = torch.from_numpy(spk_embed)
        if not torch.is_tensor(spk_embed):
            raise TypeError(f"Unsupported spk_embed type: {type(spk_embed)!r}")
        spk_embed = spk_embed.float().to(self.device)
        if spk_embed.dim() == 1:
            spk_embed = spk_embed.unsqueeze(0).unsqueeze(1)
        elif spk_embed.dim() == 2:
            spk_embed = spk_embed.unsqueeze(1)
        return spk_embed

    @staticmethod
    def _build_prompt_global_weight(
        *,
        prompt_speech_mask: torch.Tensor,
        prompt_run_stability: torch.Tensor | None,
        prompt_open_run_mask: torch.Tensor | None = None,
        drop_edge_runs: int = 0,
        allow_shape_repair: bool = False,
    ) -> torch.Tensor:
        speech = prompt_speech_mask.float().clamp(0.0, 1.0)
        stability = (
            torch.ones_like(speech)
            if not isinstance(prompt_run_stability, torch.Tensor)
            else prompt_run_stability.float().clamp(0.0, 1.0)
        )
        if tuple(stability.shape) != tuple(speech.shape):
            if not bool(allow_shape_repair):
                raise RuntimeError(
                    "prompt_run_stability shape mismatch: "
                    f"{tuple(stability.shape)} vs {tuple(speech.shape)}. "
                    "Set rhythm_v3_allow_prompt_weight_shape_repair=true only for explicit debug repair."
                )
            resized = torch.ones_like(speech)
            flat = stability.reshape(-1)
            limit = min(int(resized.numel()), int(flat.numel()))
            resized.reshape(-1)[:limit] = flat[:limit]
            stability = resized
        weight = speech * stability
        if isinstance(prompt_open_run_mask, torch.Tensor):
            open_mask = prompt_open_run_mask.float().clamp(0.0, 1.0)
            if tuple(open_mask.shape) != tuple(speech.shape):
                if not bool(allow_shape_repair):
                    raise RuntimeError(
                        "prompt_open_run_mask shape mismatch: "
                        f"{tuple(open_mask.shape)} vs {tuple(speech.shape)}. "
                        "Set rhythm_v3_allow_prompt_weight_shape_repair=true only for explicit debug repair."
                    )
                resized = torch.zeros_like(speech)
                flat = open_mask.reshape(-1)
                limit = min(int(resized.numel()), int(flat.numel()))
                resized.reshape(-1)[:limit] = flat[:limit]
                open_mask = resized
            weight = weight * (1.0 - open_mask)
        drop_edge_runs = max(0, int(drop_edge_runs))
        if drop_edge_runs > 0:
            active = torch.nonzero(weight.reshape(-1) > 0.0, as_tuple=False).reshape(-1)
            if int(active.numel()) > (2 * drop_edge_runs):
                flat_weight = weight.reshape(-1)
                flat_weight[active[:drop_edge_runs]] = 0.0
                flat_weight[active[-drop_edge_runs:]] = 0.0
                weight = flat_weight.reshape_as(weight)
        return weight

    @staticmethod
    def _extract_batch_vector(
        value,
        *,
        batch_index: int = 0,
    ):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().reshape(1)
            if value.ndim == 1:
                return value.detach()
            if value.size(0) <= batch_index:
                return None
            return value[batch_index].detach()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.reshape(1)
            if value.ndim == 1:
                return value
            if value.shape[0] <= batch_index:
                return None
            return value[batch_index]
        return None

    @staticmethod
    def _summarize_prefix_budget_abs_p95(prefix_offset) -> float | None:
        vector = StreamingVoiceConversion._extract_batch_vector(prefix_offset)
        if vector is None:
            return None
        tensor = torch.as_tensor(vector, dtype=torch.float32).reshape(-1).abs()
        if tensor.numel() <= 0:
            return None
        return float(torch.quantile(tensor, 0.95).item())

    @staticmethod
    def _summarize_rounding_residual_abs_p95(rounding_residual) -> float | None:
        vector = StreamingVoiceConversion._extract_batch_vector(rounding_residual)
        if vector is None:
            return None
        tensor = torch.as_tensor(vector, dtype=torch.float32).reshape(-1).abs()
        if tensor.numel() <= 0:
            return None
        return float(torch.quantile(tensor, 0.95).item())

    @staticmethod
    def _summarize_budget_hit_rate(budget_hit_mask) -> float | None:
        vector = StreamingVoiceConversion._extract_batch_vector(budget_hit_mask)
        if vector is None:
            return None
        tensor = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
        if tensor.numel() <= 0:
            return None
        return float((tensor > 0.5).float().mean().item())

    @staticmethod
    def _summarize_mask_sum(mask) -> float | None:
        vector = StreamingVoiceConversion._extract_batch_vector(mask)
        if vector is None:
            return None
        tensor = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
        if tensor.numel() <= 0:
            return None
        return float((tensor > 0.5).float().sum().item())

    @staticmethod
    def _assert_runtime_commit_invariants(execution) -> None:
        if execution is None:
            return
        open_tail_violation = getattr(execution, "open_tail_commit_violation", None)
        if isinstance(open_tail_violation, torch.Tensor) and bool((open_tail_violation.float() > 0.5).any().item()):
            raise RuntimeError("Open-tail units reached the committed runtime surface.")
        closed_prefix_ok = getattr(execution, "commit_closed_prefix_ok", None)
        if isinstance(closed_prefix_ok, torch.Tensor) and bool((closed_prefix_ok.float() <= 0.5).any().item()):
            raise RuntimeError("Committed runtime units must form a contiguous closed prefix.")

    @staticmethod
    def _summarize_boundary_decay_applied_rate(boundary_decay) -> float | None:
        vector = StreamingVoiceConversion._extract_batch_vector(boundary_decay)
        if vector is None:
            return None
        tensor = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
        if tensor.numel() <= 0:
            return None
        return float((tensor > 0.5).float().mean().item())

    @staticmethod
    def _resolve_uncommitted_eos_tail(
        *,
        last_mel_out: torch.Tensor,
        prev_committed_len: int,
        allow_tail_flush: bool,
    ) -> tuple[torch.Tensor, int]:
        unresolved = max(0, int(last_mel_out.shape[0]) - int(prev_committed_len))
        if unresolved <= 0:
            return last_mel_out.new_zeros((0, last_mel_out.size(-1))), 0
        mel_tail = last_mel_out[int(prev_committed_len):]
        if mel_tail.numel() <= 0:
            return last_mel_out.new_zeros((0, last_mel_out.size(-1))), 0
        if not bool(allow_tail_flush):
            raise RuntimeError(
                f"Final sealed pass left {unresolved} mel frames uncommitted; strict committed-only EOS requested."
            )
        return mel_tail, unresolved

    @staticmethod
    def _make_prompt_conditioning_cache_key(
        ref_mel_batch: torch.Tensor,
        *,
        frontend_signature: str,
        ref_source_id: str | None = None,
        ref_cache_id: str | None = None,
    ) -> str:
        digest = hashlib.sha1()
        digest.update(str(frontend_signature).encode("utf-8"))
        if ref_cache_id:
            digest.update(b"ref_cache_id:")
            digest.update(str(ref_cache_id).encode("utf-8"))
            return digest.hexdigest()
        if ref_source_id:
            ref_path = os.path.abspath(str(ref_source_id))
            if os.path.isfile(ref_path):
                digest.update(b"ref_source_path:")
                stat = os.stat(ref_path)
                digest.update(ref_path.encode("utf-8"))
                digest.update(str(int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1.0e9)))).encode("utf-8"))
                digest.update(str(int(stat.st_size)).encode("utf-8"))
                return digest.hexdigest()
        digest.update(b"ref_mel_sha1:")
        ref_cpu = ref_mel_batch.detach().to(device="cpu", dtype=torch.float32).contiguous()
        array = ref_cpu.numpy()
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _clone_prompt_conditioning_to_device(
        payload: Dict[str, torch.Tensor],
        *,
        device: str | torch.device,
    ) -> Dict[str, torch.Tensor]:
        cloned: Dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone().to(device=device)
        return cloned

    def _remember_prompt_conditioning_cache(self, key: str, payload: Dict[str, torch.Tensor]) -> None:
        if self._prompt_conditioning_cache_size <= 0:
            return
        cacheable = {
            name: tensor.detach().to(device="cpu").clone()
            for name, tensor in payload.items()
            if isinstance(tensor, torch.Tensor) and name != "prompt_spk_embed"
        }
        self._prompt_conditioning_cache[key] = cacheable
        self._prompt_conditioning_cache.move_to_end(key)
        while len(self._prompt_conditioning_cache) > self._prompt_conditioning_cache_size:
            self._prompt_conditioning_cache.popitem(last=False)

    def _extract_prompt_unit_conditioning(
        self,
        ref_mel_batch: torch.Tensor,
        prepared_spk_embed: torch.Tensor | None = None,
        ref_source_id: str | None = None,
        ref_cache_id: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        cache_key = None
        if self._prompt_conditioning_cache_size > 0:
            cache_key = self._make_prompt_conditioning_cache_key(
                ref_mel_batch,
                frontend_signature=self.rhythm_v3_frontend_signature,
                ref_source_id=ref_source_id,
                ref_cache_id=ref_cache_id,
            )
            cached = self._prompt_conditioning_cache.get(cache_key)
            if cached is not None:
                out = self._clone_prompt_conditioning_to_device(cached, device=self.device)
                if prepared_spk_embed is not None:
                    prompt_spk = prepared_spk_embed
                    if prompt_spk.dim() == 3 and prompt_spk.size(-1) == 1:
                        prompt_spk = prompt_spk.squeeze(-1)
                    elif prompt_spk.dim() == 3 and prompt_spk.size(1) == 1:
                        prompt_spk = prompt_spk.squeeze(1)
                    if prompt_spk.dim() == 2:
                        out["prompt_spk_embed"] = prompt_spk.detach()
                self._prompt_conditioning_cache.move_to_end(cache_key)
                return out
        with torch.no_grad():
            ref_lengths = torch.tensor([ref_mel_batch.size(1)], dtype=torch.long, device=ref_mel_batch.device)
            codes, _, _ = self.emformer.emformer.infer(ref_mel_batch, ref_lengths, None)
            if self.emformer.mode == "both":
                codes = self.emformer.proj1(codes)
            else:
                codes = self.emformer.proj(codes)
            if codes.dim() == 3 and codes.shape[-1] > 1:
                codes = torch.argmax(codes, dim=-1)

        cache = build_source_rhythm_cache(
            codes[0].detach().reshape(-1),
            silent_token=self.hparams.get("silent_token", 57),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            emit_silence_runs=bool(self.hparams.get("rhythm_v3_emit_silence_runs", True)),
            debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
            phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            unit_prior_path=self.hparams.get("rhythm_v3_unit_prior_path"),
        )
        assert_duration_v3_cache_meta_compatible(
            cache.get(DURATION_V3_CACHE_META_KEY),
            silent_token=int(self.hparams.get("silent_token", 57)),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            emit_silence_runs=bool(self.hparams.get("rhythm_v3_emit_silence_runs", True)),
            debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
            phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        )
        prompt_content_units = torch.tensor(cache["content_units"], device=self.device, dtype=torch.long).unsqueeze(0)
        prompt_duration_obs = torch.tensor(cache["dur_anchor_src"], device=self.device, dtype=torch.float32).unsqueeze(0)
        prompt_valid_mask = (prompt_duration_obs > 0).float()
        prompt_silence = cache.get("source_silence_mask")
        minimal_v1_profile = bool(self.hparams.get("rhythm_v3_minimal_v1_profile", False))
        g_variant = normalize_global_rate_variant(self.hparams.get("rhythm_v3_g_variant", "raw_median"))
        if prompt_silence is not None:
            prompt_silence_mask = torch.tensor(prompt_silence, device=self.device, dtype=torch.float32).unsqueeze(0)
            prompt_speech_mask = prompt_valid_mask * (1.0 - prompt_silence_mask.clamp(0.0, 1.0))
        else:
            if minimal_v1_profile:
                raise ValueError(
                    "rhythm_v3_minimal_v1_profile requires cached prompt source_silence_mask "
                    "to preserve speech-only global tempo statistics."
                )
            prompt_silence_mask = None
            prompt_speech_mask = prompt_valid_mask
        out = {
            "prompt_content_units": prompt_content_units,
            "prompt_duration_obs": prompt_duration_obs,
            "prompt_unit_mask": prompt_valid_mask,
            "prompt_valid_mask": prompt_valid_mask,
            "prompt_speech_mask": prompt_speech_mask,
            "prompt_source_boundary_cue": torch.tensor(
                cache.get("source_boundary_cue", np.zeros_like(cache["dur_anchor_src"], dtype=np.float32)),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0),
            "prompt_phrase_group_pos": torch.tensor(
                cache.get("phrase_group_pos", np.zeros_like(cache["dur_anchor_src"], dtype=np.float32)),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0),
            "prompt_phrase_final_mask": torch.tensor(
                cache.get("phrase_final_mask", np.zeros_like(cache["dur_anchor_src"], dtype=np.float32)),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0),
        }
        prompt_run_stability = torch.tensor(
            cache.get("source_run_stability", np.ones_like(cache["dur_anchor_src"], dtype=np.float32)),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        prompt_open_run_mask = torch.tensor(
            cache.get("open_run_mask", np.zeros_like(cache["dur_anchor_src"], dtype=np.float32)),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        out["prompt_global_weight"] = self._build_prompt_global_weight(
            prompt_speech_mask=prompt_speech_mask,
            prompt_run_stability=prompt_run_stability,
            prompt_open_run_mask=prompt_open_run_mask,
            drop_edge_runs=int(self.hparams.get("rhythm_v3_drop_edge_runs_for_g", 0) or 0),
            allow_shape_repair=bool(self.hparams.get("rhythm_v3_allow_prompt_weight_shape_repair", False)),
        )
        out["prompt_global_weight_present"] = torch.ones((1, 1), device=self.device, dtype=torch.float32)
        valid_mass = prompt_valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        speech_mass = prompt_speech_mask.sum(dim=1, keepdim=True)
        out["prompt_speech_ratio_scalar"] = (speech_mass / valid_mass).to(dtype=torch.float32)
        hop_size = float(self.hparams.get("hop_size", 0.0) or 0.0)
        sample_rate = float(self.hparams.get("audio_sample_rate", 0.0) or 0.0)
        if hop_size > 0.0 and sample_rate > 0.0:
            total_frames = float(prompt_duration_obs.sum().item())
            out["prompt_ref_len_sec"] = torch.full(
                (1, 1),
                total_frames * hop_size / sample_rate,
                device=self.device,
                dtype=torch.float32,
            )
        out["g_trim_ratio"] = torch.full(
            (1, 1),
            float(self.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2),
            device=self.device,
            dtype=torch.float32,
        )
        prompt_unit_log_prior = cache.get("prompt_unit_log_prior")
        if prompt_unit_log_prior is None:
            prompt_unit_log_prior = cache.get("unit_log_prior")
        unit_prior_meta = cache.get(UNIT_LOG_PRIOR_META_KEY, {})
        if prompt_unit_log_prior is not None:
            prompt_unit_log_prior_arr = np.asarray(prompt_unit_log_prior, dtype=np.float32).reshape(-1)
            if prompt_unit_log_prior_arr.shape[0] != int(prompt_duration_obs.shape[1]):
                raise RuntimeError(
                    "prompt_unit_log_prior must match prompt run shape for maintained prompt conditioning: "
                    f"{tuple(prompt_unit_log_prior_arr.shape)} vs {(int(prompt_duration_obs.shape[1])),}"
                )
            out["prompt_unit_log_prior"] = torch.tensor(
                prompt_unit_log_prior_arr,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            out["prompt_unit_log_prior_present"] = torch.ones((1, 1), device=self.device, dtype=torch.float32)
            out["prompt_unit_prior_vocab_size"] = torch.tensor(
                [[int(unit_prior_meta.get("unit_prior_vocab_size", 0) or 0)]],
                device=self.device,
                dtype=torch.long,
            )
        elif g_variant == "unit_norm":
            raise RuntimeError(
                "rhythm_v3 g_variant=unit_norm requires prompt_unit_log_prior/unit_log_prior "
                "matching prompt runs in prompt conditioning."
            )
        else:
            out["prompt_unit_log_prior_present"] = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
            out["prompt_unit_prior_vocab_size"] = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        if prompt_silence is not None:
            out["prompt_silence_mask"] = prompt_silence_mask
        rhythm_frontend = getattr(self.model, "rhythm_unit_frontend", None)
        rhythm_module = getattr(self.model, "rhythm_module", None)
        rate_mode = str(getattr(rhythm_module, "rate_mode", "") or "").strip().lower()
        if rhythm_frontend is not None:
            if rate_mode == "simple_global":
                out["prompt_log_base"] = torch.zeros_like(prompt_duration_obs)
                out["prompt_unit_anchor_base"] = prompt_duration_obs.clamp_min(1.0e-4)
            else:
                out["prompt_log_base"] = rhythm_frontend.compute_rate_log_base(
                    prompt_content_units,
                    prompt_valid_mask,
                    stop_gradient=True,
                ).detach()
        if prepared_spk_embed is not None:
            prompt_spk = prepared_spk_embed
            if prompt_spk.dim() == 3 and prompt_spk.size(-1) == 1:
                prompt_spk = prompt_spk.squeeze(-1)
            elif prompt_spk.dim() == 3 and prompt_spk.size(1) == 1:
                prompt_spk = prompt_spk.squeeze(1)
            if prompt_spk.dim() == 2:
                out["prompt_spk_embed"] = prompt_spk.detach()
        if cache_key is not None:
            self._remember_prompt_conditioning_cache(cache_key, out)
        return out

    def _render_vocoder_chunk(
        self,
        mel_chunk: torch.Tensor,
        *,
        mel_context_buffer: RollingMelContextBuffer,
        f0_chunk=None,
    ) -> np.ndarray:
        if mel_chunk.numel() <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.vocoder_native_streaming_capable:
            return np.asarray(
                self.vocoder.spec2wav_stream(mel_chunk.detach().cpu().numpy(), f0=f0_chunk),
                dtype=np.float32,
            )

        mel_window, context_frames = mel_context_buffer.build_window(mel_chunk)
        wav_window = np.asarray(
            self.vocoder.spec2wav(mel_window.detach().cpu().numpy(), f0=f0_chunk),
            dtype=np.float32,
        )
        hop_size = int(self.hparams["hop_size"])
        new_frames = int(mel_chunk.size(0))
        start_sample = max(0, context_frames * hop_size)
        end_sample = min(len(wav_window), start_sample + new_frames * hop_size)
        return wav_window[start_sample:end_sample]

    def infer_once(
        self,
        inp: Dict,
        spk_embed=None,
        return_metadata: bool = False,
        return_debug_bundle: bool = False,
        ref_cache_id: str | None = None,
    ):
        if hasattr(self.vocoder, "reset_stream"):
            self.vocoder.reset_stream()
            self._vocoder_warm_zero()

        ref_mel_np = self._wav_to_mel(inp["ref_wav"])
        ref_mel = torch.from_numpy(ref_mel_np).float().to(self.device)

        src_mel_np = self._wav_to_mel(inp["src_wav"])
        src_mel = torch.from_numpy(src_mel_np).unsqueeze(0).to(self.device)  # [1, T, 80]
        total_frames = int(src_mel.shape[1])

        seg = resolve_chunk_frames(self.hparams)
        rc = int(self.hparams["right_context"])
        mel_frame_ms = resolve_mel_frame_ms(self.hparams)

        code_buffer = PrefixCodeBuffer(total_capacity=max(total_frames, 1))
        mel_context_buffer = RollingMelContextBuffer(
            left_context_frames=self.vocoder_left_context_frames
        )
        mel_chunks: list[torch.Tensor] = []
        wav_chunks: list[np.ndarray] = []
        prev_committed_len = 0
        pos = 0
        num_chunks = 0
        state = None
        rhythm_state = None
        rhythm_source_cache = None
        rhythm_frontend = getattr(self.model, "rhythm_unit_frontend", None)
        rhythm_unitizer_state = None
        decoder_cache = None
        rhythm_ref_conditioning = None
        last_mel_out = None
        eos_tail_closed = False
        content_window_left_tokens = self._resolve_content_window_left_tokens()
        committed_token_frontier = 0
        previous_resolved_frontier = None
        max_content_window_start = 0
        final_debug_bundle = None
        last_projector_prefix_offset = None
        last_projector_prefix_drift = None
        last_projector_rounding_residual = None
        last_projector_budget_hit_mask = None
        last_projector_boundary_decay = None
        last_open_tail_commit_violation = None
        last_commit_closed_prefix_ok = None
        allow_eos_tail_flush_fallback = bool(
            self.hparams.get(
                "rhythm_allow_eos_tail_flush_fallback",
                True,
            )
        )
        prompt_sidecar_global_weight_present = False
        prompt_sidecar_unit_log_prior_present = False
        g_variant = normalize_global_rate_variant(self.hparams.get("rhythm_v3_g_variant", "raw_median"))
        effective_ref_cache_id = ref_cache_id if ref_cache_id is not None else inp.get("ref_cache_id")

        ref_mel_batch = ref_mel.unsqueeze(0)
        with torch.no_grad():
            prepared_spk_embed = self._prepare_spk_embed(ref_mel_batch, spk_embed=spk_embed)
            if getattr(
                self.model,
                "rhythm_enabled",
                bool(
                    getattr(self.model, "rhythm_enable_v2", False)
                    or getattr(self.model, "rhythm_enable_v3", False)
                ),
            ):
                if getattr(self.model, "rhythm_enable_v3", False):
                    rhythm_ref_conditioning = self._extract_prompt_unit_conditioning(
                        ref_mel_batch,
                        prepared_spk_embed=prepared_spk_embed,
                        ref_source_id=inp.get("ref_wav"),
                        ref_cache_id=effective_ref_cache_id,
                    )
                    prompt_sidecar_global_weight_present = bool(
                        torch.as_tensor(
                            rhythm_ref_conditioning.get("prompt_global_weight_present", 0.0),
                            dtype=torch.float32,
                        ).reshape(-1)[:1].gt(0.5).any().item()
                    )
                    prompt_sidecar_unit_log_prior_present = bool(
                        torch.as_tensor(
                            rhythm_ref_conditioning.get("prompt_unit_log_prior_present", 0.0),
                            dtype=torch.float32,
                        ).reshape(-1)[:1].gt(0.5).any().item()
                    )
                else:
                    rhythm_ref_conditioning = self.model.prepare_rhythm_reference(
                        ref_mel_batch,
                        ref_lengths=torch.tensor([ref_mel_batch.size(1)], device=ref_mel_batch.device),
                    )
        require_runtime_ref = bool(self.hparams.get("style", False)) and not bool(
            getattr(self.model, "rhythm_minimal_style_only", False)
        )

        while pos < total_frames:
            num_chunks += 1
            emit = min(seg, total_frames - pos)
            look = min(rc, total_frames - (pos + emit))

            real_len = emit + look
            chunk = src_mel[:, pos : pos + real_len, :]
            need_pad = (seg + rc) - real_len
            if need_pad > 0:
                pad = chunk[:, -1:, :].expand(1, need_pad, src_mel.shape[2])
                chunk = torch.cat([chunk, pad], dim=1)

            lengths = torch.full((1,), chunk.size(1), dtype=torch.long, device=self.device)
            with torch.no_grad():
                chunk_out, _, state = self.emformer.emformer.infer(chunk, lengths, state)
                if self.emformer.mode == "both":
                    chunk_out = self.emformer.proj1(chunk_out)
                else:
                    chunk_out = self.emformer.proj(chunk_out)
                if chunk_out.dim() == 3 and chunk_out.shape[-1] > 1:
                    chunk_out = torch.argmax(chunk_out, dim=-1)

            new_codes = chunk_out[:, :emit]
            all_codes = code_buffer.append(new_codes)
            if rhythm_frontend is not None:
                if rhythm_unitizer_state is None:
                    rhythm_unitizer_state = rhythm_frontend.init_stream_state(
                        batch_size=1,
                        device=self.device,
                    )
                unit_batch, rhythm_unitizer_state = rhythm_frontend.step_content_tensor(
                    new_codes,
                    state=rhythm_unitizer_state,
                    content_lengths=torch.tensor([new_codes.size(1)], dtype=torch.long, device=self.device),
                    mark_last_open=True,
                )
                rhythm_source_cache = {
                    key: value
                    for key, value in export_duration_v3_source_cache(unit_batch).items()
                    if value is not None
                }
                rhythm_source_cache[DURATION_V3_CACHE_META_KEY] = dict(self.rhythm_v3_cache_meta)
                cache_total = int(rhythm_source_cache["source_duration_obs"].sum().item())
                if cache_total != int(all_codes.size(1)):
                    raise RuntimeError(
                        f"Incremental rhythm source cache drifted from all_codes: cache_total={cache_total}, all_codes={int(all_codes.size(1))}"
                    )

            with torch.no_grad():
                model_content = all_codes
                model_content_offset = self._compute_content_window_start(
                    total_tokens=int(all_codes.size(1)),
                    committed_frontier_tokens=int(committed_token_frontier),
                    left_context_tokens=int(content_window_left_tokens),
                )
                if model_content_offset > 0:
                    model_content = all_codes[:, model_content_offset:]
                max_content_window_start = max(max_content_window_start, int(model_content_offset))
                out = self.model(
                    content=model_content,
                    spk_embed=prepared_spk_embed,
                    target=None,
                    ref=ref_mel_batch if require_runtime_ref else None,
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=200000,
                    content_lengths=torch.tensor([model_content.size(1)], device=self.device),
                    rhythm_state=rhythm_state,
                    rhythm_ref_conditioning=rhythm_ref_conditioning,
                    rhythm_source_cache=rhythm_source_cache,
                    decoder_cache=decoder_cache,
                )
                next_rhythm_state = out.get("rhythm_state_next", rhythm_state)
                self._assert_committed_prefix_not_rewritten(
                    prev_state=rhythm_state,
                    next_state=next_rhythm_state,
                )
                rhythm_state = next_rhythm_state
                execution = out.get("rhythm_execution")
                if execution is not None:
                    self._assert_runtime_commit_invariants(execution)
                    last_projector_prefix_offset = self._extract_batch_vector(
                        getattr(execution, "prefix_unit_offset", None),
                        batch_index=0,
                    )
                    last_projector_prefix_drift = self._extract_batch_vector(
                        getattr(execution, "projector_prefix_drift", None),
                        batch_index=0,
                    )
                    last_projector_rounding_residual = self._extract_batch_vector(
                        getattr(execution, "projector_rounding_residual", None),
                        batch_index=0,
                    )
                    last_projector_budget_hit_mask = self._extract_batch_vector(
                        getattr(execution, "projector_budget_hit_mask", None),
                        batch_index=0,
                    )
                    last_projector_boundary_decay = self._extract_batch_vector(
                        getattr(execution, "projector_boundary_decay_applied", None),
                        batch_index=0,
                    )
                    last_open_tail_commit_violation = self._extract_batch_vector(
                        getattr(execution, "open_tail_commit_violation", None),
                        batch_index=0,
                    )
                    last_commit_closed_prefix_ok = self._extract_batch_vector(
                        getattr(execution, "commit_closed_prefix_ok", None),
                        batch_index=0,
                    )
                rhythm_ref_conditioning = out.get("rhythm_ref_conditioning", rhythm_ref_conditioning)
                decoder_cache = out.get("decoder_cache", decoder_cache)
                last_mel_out = out["mel_out"][0]
                resolved_committed_frontier = self._resolve_committed_token_frontier_from_cache(
                    commit_frontier=out.get("commit_frontier"),
                    rhythm_source_cache=rhythm_source_cache,
                    batch_index=0,
                )
                if isinstance(resolved_committed_frontier, int):
                    self._assert_monotone_committed_frontier(
                        previous_frontier=previous_resolved_frontier,
                        current_frontier=resolved_committed_frontier,
                    )
                    previous_resolved_frontier = resolved_committed_frontier
                    committed_token_frontier = max(committed_token_frontier, resolved_committed_frontier)
                if return_debug_bundle:
                    final_debug_bundle = build_debug_record(
                        model_output=out,
                        metadata={
                            "phase": "inference",
                            "src_wav": inp.get("src_wav"),
                            "ref_wav": inp.get("ref_wav"),
                            "ref_condition": "real",
                            "streaming_chunk_index": int(num_chunks),
                        },
                    ).to_dict()

            mel_new, prev_committed_len = extract_incremental_committed_mel(
                out,
                prev_committed_len=prev_committed_len,
                batch_index=0,
            )
            if mel_new.numel() > 0:
                mel_chunks.append(mel_new)
                wav_chunk = self._render_vocoder_chunk(
                    mel_new,
                    mel_context_buffer=mel_context_buffer,
                )
                if len(wav_chunk) > 0:
                    wav_chunks.append(wav_chunk)
            pos += emit

        if last_mel_out is None:
            raise RuntimeError("Streaming inference produced no mel output.")

        if (
            prev_committed_len < int(last_mel_out.shape[0])
            and isinstance(rhythm_source_cache, dict)
            and isinstance(rhythm_source_cache.get("unit_mask"), torch.Tensor)
        ):
            final_cache = {
                key: (value.clone() if isinstance(value, torch.Tensor) else value)
                for key, value in rhythm_source_cache.items()
            }
            final_cache["sealed_mask"] = final_cache["unit_mask"].clone().float()
            final_content = all_codes
            final_content_offset = self._compute_content_window_start(
                total_tokens=int(all_codes.size(1)),
                committed_frontier_tokens=int(committed_token_frontier),
                left_context_tokens=int(content_window_left_tokens),
            )
            if final_content_offset > 0:
                final_content = all_codes[:, final_content_offset:]
            max_content_window_start = max(max_content_window_start, int(final_content_offset))
            with torch.no_grad():
                final_out = self.model(
                    content=final_content,
                    spk_embed=prepared_spk_embed,
                    target=None,
                    ref=ref_mel_batch if require_runtime_ref else None,
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=200000,
                    content_lengths=torch.tensor([final_content.size(1)], device=self.device),
                    rhythm_state=rhythm_state,
                    rhythm_ref_conditioning=rhythm_ref_conditioning,
                    rhythm_source_cache=final_cache,
                    decoder_cache=decoder_cache,
                )
            next_rhythm_state = final_out.get("rhythm_state_next", rhythm_state)
            self._assert_committed_prefix_not_rewritten(
                prev_state=rhythm_state,
                next_state=next_rhythm_state,
            )
            rhythm_state = next_rhythm_state
            execution = final_out.get("rhythm_execution")
            if execution is not None:
                self._assert_runtime_commit_invariants(execution)
                last_projector_prefix_offset = self._extract_batch_vector(
                    getattr(execution, "prefix_unit_offset", None),
                    batch_index=0,
                )
                last_projector_prefix_drift = self._extract_batch_vector(
                    getattr(execution, "projector_prefix_drift", None),
                    batch_index=0,
                )
                last_projector_rounding_residual = self._extract_batch_vector(
                    getattr(execution, "projector_rounding_residual", None),
                    batch_index=0,
                )
                last_projector_budget_hit_mask = self._extract_batch_vector(
                    getattr(execution, "projector_budget_hit_mask", None),
                    batch_index=0,
                )
                last_projector_boundary_decay = self._extract_batch_vector(
                    getattr(execution, "projector_boundary_decay_applied", None),
                    batch_index=0,
                )
                last_open_tail_commit_violation = self._extract_batch_vector(
                    getattr(execution, "open_tail_commit_violation", None),
                    batch_index=0,
                )
                last_commit_closed_prefix_ok = self._extract_batch_vector(
                    getattr(execution, "commit_closed_prefix_ok", None),
                    batch_index=0,
                )
            decoder_cache = final_out.get("decoder_cache", decoder_cache)
            last_mel_out = final_out["mel_out"][0]
            final_committed_frontier = self._resolve_committed_token_frontier_from_cache(
                commit_frontier=final_out.get("commit_frontier"),
                rhythm_source_cache=rhythm_source_cache,
                batch_index=0,
            )
            if isinstance(final_committed_frontier, int):
                self._assert_monotone_committed_frontier(
                    previous_frontier=previous_resolved_frontier,
                    current_frontier=final_committed_frontier,
                )
                previous_resolved_frontier = final_committed_frontier
                committed_token_frontier = max(committed_token_frontier, final_committed_frontier)
            if return_debug_bundle:
                final_debug_bundle = build_debug_record(
                    model_output=final_out,
                    metadata={
                        "phase": "inference",
                        "src_wav": inp.get("src_wav"),
                        "ref_wav": inp.get("ref_wav"),
                        "ref_condition": "real",
                        "streaming_chunk_index": int(num_chunks),
                        "eos_tail_closed": True,
                    },
                ).to_dict()
            final_mel_new, prev_committed_len = extract_incremental_committed_mel(
                final_out,
                prev_committed_len=prev_committed_len,
                batch_index=0,
            )
            if final_mel_new.numel() > 0:
                eos_tail_closed = True
                mel_chunks.append(final_mel_new)
                wav_tail = self._render_vocoder_chunk(
                    final_mel_new,
                    mel_context_buffer=mel_context_buffer,
                )
                if len(wav_tail) > 0:
                    wav_chunks.append(wav_tail)

        unresolved_eos_tail_frames = 0
        if prev_committed_len < int(last_mel_out.shape[0]):
            mel_tail, unresolved_eos_tail_frames = self._resolve_uncommitted_eos_tail(
                last_mel_out=last_mel_out,
                prev_committed_len=prev_committed_len,
                allow_tail_flush=allow_eos_tail_flush_fallback,
            )
            if mel_tail.numel() > 0:
                mel_chunks.append(mel_tail)
                wav_tail = self._render_vocoder_chunk(
                    mel_tail,
                    mel_context_buffer=mel_context_buffer,
                )
                if len(wav_tail) > 0:
                    wav_chunks.append(wav_tail)

        if mel_chunks:
            mel_pred = torch.cat(mel_chunks, dim=0)
        else:
            mel_pred = last_mel_out.new_zeros((0, last_mel_out.size(-1)))

        if wav_chunks:
            wav_pred = np.concatenate(wav_chunks, axis=0)
        elif mel_pred.numel() > 0:
            wav_pred = self.vocoder.spec2wav(mel_pred.detach().cpu().numpy())
        else:
            wav_pred = np.zeros(0, dtype=np.float32)

        duration_seconds = float(total_frames * self.hparams["hop_size"]) / float(
            self.hparams["audio_sample_rate"]
        )
        self.last_inference_metadata = self.get_streaming_runtime_metadata(
            duration_seconds=duration_seconds
        )
        if last_projector_prefix_offset is None and isinstance(final_debug_bundle, dict):
            last_projector_prefix_offset = final_debug_bundle.get("prefix_unit_offset")
        if last_projector_prefix_drift is None:
            last_projector_prefix_drift = (
                final_debug_bundle.get("projector_prefix_drift")
                if isinstance(final_debug_bundle, dict)
                else None
            )
        if last_projector_prefix_drift is None:
            last_projector_prefix_drift = last_projector_prefix_offset
        if last_projector_rounding_residual is None and isinstance(final_debug_bundle, dict):
            last_projector_rounding_residual = final_debug_bundle.get("projector_rounding_residual")
        if last_projector_budget_hit_mask is None and isinstance(final_debug_bundle, dict):
            last_projector_budget_hit_mask = final_debug_bundle.get("projector_budget_hit_mask")
        if last_projector_boundary_decay is None and isinstance(final_debug_bundle, dict):
            last_projector_boundary_decay = final_debug_bundle.get("projector_boundary_decay_applied")
        if last_open_tail_commit_violation is None and isinstance(final_debug_bundle, dict):
            last_open_tail_commit_violation = final_debug_bundle.get("open_tail_commit_violation")
        if last_commit_closed_prefix_ok is None and isinstance(final_debug_bundle, dict):
            last_commit_closed_prefix_ok = final_debug_bundle.get("commit_closed_prefix_ok")
        eos_tail_flush_severity = "none"
        if unresolved_eos_tail_frames > 0:
            eos_tail_flush_severity = "warning" if allow_eos_tail_flush_fallback else "error"
        self.last_inference_metadata.update(
            {
                "source_total_frames": int(total_frames),
                "source_duration_seconds": float(duration_seconds),
                "num_streaming_chunks": int(num_chunks),
                "committed_mel_frames": int(mel_pred.shape[0]),
                "mel_frame_ms": float(mel_frame_ms),
                "vocoder_left_context_frames_effective": int(self.vocoder_left_context_frames),
                "vocoder_left_context_source": str(self.vocoder_left_context_source),
                "vocoder_native_streaming_capable": bool(self.vocoder_native_streaming_capable),
                "emformer_stateful_content_frontend": True,
                "rhythm_incremental_source_cache": True,
                "rhythm_final_tail_closed_eos": bool(eos_tail_closed),
                "rhythm_final_tail_is_not_strictly_committed_only": bool(unresolved_eos_tail_frames > 0),
                "rhythm_uncommitted_eos_tail_frames": int(unresolved_eos_tail_frames),
                "rhythm_allow_eos_tail_flush_fallback": bool(allow_eos_tail_flush_fallback),
                "rhythm_eos_tail_flush_severity": str(eos_tail_flush_severity),
                "prompt_global_weight_present": bool(prompt_sidecar_global_weight_present),
                "prompt_unit_log_prior_present": bool(prompt_sidecar_unit_log_prior_present),
                "rhythm_v3_g_variant": str(g_variant),
                "rhythm_prefix_budget_abs_p95": self._summarize_prefix_budget_abs_p95(last_projector_prefix_offset),
                "rhythm_prefix_drift_abs_p95": self._summarize_prefix_budget_abs_p95(last_projector_prefix_drift),
                "rhythm_rounding_residual_abs_p95": self._summarize_rounding_residual_abs_p95(last_projector_rounding_residual),
                "rhythm_budget_hit_rate": self._summarize_budget_hit_rate(last_projector_budget_hit_mask),
                "rhythm_boundary_decay_applied_rate": self._summarize_boundary_decay_applied_rate(last_projector_boundary_decay),
                "rhythm_open_tail_commit_violation_count": self._summarize_mask_sum(last_open_tail_commit_violation),
                "rhythm_committed_closed_prefix_ok": (
                    None
                    if last_commit_closed_prefix_ok is None
                    else bool(
                        torch.as_tensor(last_commit_closed_prefix_ok, dtype=torch.float32)
                        .reshape(-1)
                        .gt(0.5)
                        .all()
                        .item()
                    )
                ),
                "content_history_windowing_enabled": True,
                "content_history_left_context_tokens": int(content_window_left_tokens),
                "content_history_max_trimmed_tokens": int(max_content_window_start),
                "content_history_last_committed_token_frontier": int(committed_token_frontier),
                "full_end_to_end_stateful_streaming": False,
            }
        )
        self.last_rhythm_debug_bundle = final_debug_bundle

        if return_metadata and return_debug_bundle:
            return wav_pred, mel_pred.cpu().numpy(), self.last_inference_metadata, final_debug_bundle
        if return_metadata:
            return wav_pred, mel_pred.cpu().numpy(), self.last_inference_metadata
        if return_debug_bundle:
            return wav_pred, mel_pred.cpu().numpy(), final_debug_bundle
        return wav_pred, mel_pred.cpu().numpy()

    def test_multiple_sentences(self, test_cases: List[Dict]):
        os.makedirs("infer_out_demo", exist_ok=True)
        for i, inp in enumerate(test_cases):
            wav, mel = self.infer_once(inp)
            ref_name = os.path.splitext(os.path.basename(inp["ref_wav"]))[0]
            src_name = os.path.splitext(os.path.basename(inp["src_wav"]))[0]
            save_path = f"infer_out_demo/{ref_name}_{src_name}.wav"
            save_wav(wav, save_path, self.hparams["audio_sample_rate"])
            print(f"Saved output: {save_path}")


if __name__ == "__main__":
    set_hparams()
    demo = [
        {"ref_wav": "path/to/reference_audio.wav", "src_wav": "path/to/source_audio.wav"},
    ]

    engine = StreamingVoiceConversion(hparams)
    engine.test_multiple_sentences(demo)
