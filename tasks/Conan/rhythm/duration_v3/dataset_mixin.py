from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from modules.Conan.rhythm.policy import is_duration_operator_mode
from modules.Conan.rhythm_v3.g_stats import normalize_global_rate_variant
from modules.Conan.rhythm_v3.source_cache import (
    UNIT_LOG_PRIOR_META_KEY,
    build_source_rhythm_cache_v3 as build_source_rhythm_cache,
    duration_v3_cache_meta_signature,
    maybe_attach_unit_log_prior_from_path,
)
from tasks.Conan.rhythm.duration_v3.alignment_projection import (
    align_target_runs_to_source_discrete as _align_target_runs_to_source_discrete,
    as_float32_1d as _as_float32_1d,
    as_int64_1d as _as_int64_1d,
    project_target_runs_onto_source as _project_target_runs_onto_source,
    resolve_run_silence_mask as _resolve_run_silence_mask,
)
from tasks.Conan.rhythm.duration_v3.targets import build_pseudo_source_duration_context
from tasks.Conan.rhythm.duration_v3.task_config import is_duration_v3_prompt_summary_backbone

_ALIGNMENT_KIND_DISCRETE = "discrete"
_ALIGNMENT_KIND_CONTINUOUS_PRECOMPUTED = "continuous_precomputed"
_ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1 = "continuous_viterbi_v1"
_ALIGNMENT_MODE_ID_CONTINUOUS_PRECOMPUTED = 1
_ALIGNMENT_MODE_ID_CONTINUOUS_VITERBI_V1 = 2
_PROMPT_WEIGHT_REPAIR_FLAG = "rhythm_v3_allow_prompt_weight_shape_repair"
_ALIGNMENT_PROVENANCE_FLOAT_KEYS = (
    "unit_alignment_band_ratio_tgt",
    "unit_alignment_lambda_emb_tgt",
    "unit_alignment_lambda_type_tgt",
    "unit_alignment_lambda_band_tgt",
    "unit_alignment_lambda_unit_tgt",
    "unit_alignment_bad_cost_threshold_tgt",
    "unit_alignment_allow_source_skip_tgt",
    "unit_alignment_skip_penalty_tgt",
)
_ALIGNMENT_PROVENANCE_OBJECT_KEYS = (
    "unit_alignment_source_cache_signature_tgt",
    "unit_alignment_target_cache_signature_tgt",
    "unit_alignment_sidecar_signature_tgt",
)
_MINIMAL_CONTINUOUS_QUALITY_KEYS = (
    "unit_alignment_unmatched_speech_ratio_tgt",
    "unit_alignment_mean_local_confidence_speech_tgt",
    "unit_alignment_mean_coarse_confidence_speech_tgt",
)
_MINIMAL_CONTINUOUS_SIGNATURE_KEYS = (
    "unit_alignment_source_cache_signature_tgt",
    "unit_alignment_target_cache_signature_tgt",
    "unit_alignment_sidecar_signature_tgt",
)
_ALIGNMENT_PROVENANCE_VITERBI_DEFAULTS = {
    "unit_alignment_band_ratio_tgt": 0.08,
    "unit_alignment_lambda_emb_tgt": 1.0,
    "unit_alignment_lambda_type_tgt": 0.5,
    "unit_alignment_lambda_band_tgt": 0.2,
    "unit_alignment_lambda_unit_tgt": 0.0,
    "unit_alignment_bad_cost_threshold_tgt": 1.2,
    "unit_alignment_allow_source_skip_tgt": 0.0,
    "unit_alignment_skip_penalty_tgt": 1.0,
}


def _extract_object_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.dtype == object:
            if value.size == 1:
                return value.reshape(-1)[0]
            if value.shape[0] == 1:
                first = value[0]
                if isinstance(first, np.ndarray):
                    first_list = first.tolist()
                    return tuple(first_list) if isinstance(first_list, list) else first_list
                return first
        flat = value.reshape(-1)
        if flat.size > 0:
            return flat[0].item() if hasattr(flat[0], "item") else flat[0]
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        return tuple(value)
    return value


class DurationV3DatasetMixin:
    def _resolve_hparam_alias(self, *keys: str, default=None):
        for key in keys:
            if not key:
                continue
            value = self.hparams.get(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def _normalize_alignment_kind_export(value, *, mode_id=None) -> str:
        normalized = str(_extract_object_scalar(value) or "").strip().lower()
        if normalized:
            return normalized
        try:
            resolved_mode_id = int(_extract_object_scalar(mode_id)) if mode_id is not None else None
        except Exception:
            resolved_mode_id = None
        if resolved_mode_id == _ALIGNMENT_MODE_ID_CONTINUOUS_PRECOMPUTED:
            return _ALIGNMENT_KIND_CONTINUOUS_PRECOMPUTED
        if resolved_mode_id == _ALIGNMENT_MODE_ID_CONTINUOUS_VITERBI_V1:
            return _ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1
        return _ALIGNMENT_KIND_DISCRETE

    @staticmethod
    def _extract_optional_float_scalar(value) -> float | None:
        scalar = _extract_object_scalar(value)
        if scalar is None:
            return None
        if isinstance(scalar, str) and scalar.strip() == "":
            return None
        try:
            return float(scalar)
        except Exception:
            return None

    @staticmethod
    def _is_continuous_alignment_kind(value, *, mode_id=None) -> bool:
        return DurationV3DatasetMixin._normalize_alignment_kind_export(value, mode_id=mode_id).startswith(
            "continuous"
        )

    def _alignment_quality_thresholds(self) -> tuple[float, float, float]:
        return (
            float(
                self._resolve_hparam_alias(
                    "rhythm_v3_alignment_unmatched_speech_ratio_max",
                    "rhythm_v3_align_unmatched_speech_ratio_max",
                    default=0.15,
                )
                or 0.15
            ),
            float(
                self._resolve_hparam_alias(
                    "rhythm_v3_alignment_mean_local_confidence_speech_min",
                    "rhythm_v3_align_mean_local_confidence_speech_min",
                    default=0.55,
                )
                or 0.55
            ),
            float(
                self._resolve_hparam_alias(
                    "rhythm_v3_alignment_mean_coarse_confidence_speech_min",
                    "rhythm_v3_align_mean_coarse_confidence_speech_min",
                    default=0.60,
                )
                or 0.60
            ),
        )

    @staticmethod
    def _normalize_signature_text(value) -> str:
        scalar = _extract_object_scalar(value)
        text = str(scalar or "").strip()
        return text or "missing"

    @staticmethod
    def _build_alignment_sidecar_signature(
        *,
        source_cache_meta_signature: str,
        target_cache_meta_signature: str,
        source_frame_states=None,
        source_frame_to_run=None,
        target_frame_states=None,
        target_frame_speech_prob=None,
        target_frame_weight=None,
        target_frame_valid=None,
        target_frame_unit_hint=None,
    ) -> str:
        payload = {
            "source_cache_meta_signature": str(source_cache_meta_signature or "missing"),
            "target_cache_meta_signature": str(target_cache_meta_signature or "missing"),
        }
        for key, value in (
            ("source_frame_states", source_frame_states),
            ("source_frame_to_run", source_frame_to_run),
            ("target_frame_states", target_frame_states),
            ("target_frame_speech_prob", target_frame_speech_prob),
            ("target_frame_weight", target_frame_weight),
            ("target_frame_valid", target_frame_valid),
            ("target_frame_unit_hint", target_frame_unit_hint),
        ):
            if value is None:
                payload[key] = None
                continue
            arr = np.asarray(value)
            arr_bytes = np.ascontiguousarray(arr).view(np.uint8)
            payload[key] = {
                "shape": tuple(int(dim) for dim in arr.shape),
                "dtype": str(arr.dtype),
                "digest": hashlib.sha1(arr_bytes.tobytes()).hexdigest(),
            }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _validate_continuous_projection_provenance(
        self,
        *,
        source_cache: dict,
        paired_target_conditioning: dict,
        context: str,
    ) -> None:
        if not isinstance(paired_target_conditioning, dict):
            return
        source_cache_signature = duration_v3_cache_meta_signature(source_cache)
        paired_source_signature = self._normalize_signature_text(
            paired_target_conditioning.get("source_cache_meta_signature")
        )
        paired_target_signature = self._normalize_signature_text(
            paired_target_conditioning.get("paired_target_cache_meta_signature")
        )
        if (
            source_cache_signature != "missing"
            and paired_source_signature != "missing"
            and paired_source_signature != source_cache_signature
        ):
            raise RuntimeError(
                "continuous paired-target source frontend/cache signature mismatch: "
                f"source_cache={source_cache_signature}, conditioning_source={paired_source_signature} ({context})"
            )
        if (
            source_cache_signature != "missing"
            and paired_target_signature != "missing"
            and paired_target_signature != source_cache_signature
        ):
            raise RuntimeError(
                "continuous paired-target source/target frontend/cache signature mismatch: "
                f"source_cache={source_cache_signature}, paired_target={paired_target_signature} ({context})"
            )

    def _enforce_minimal_continuous_alignment_gate(
        self,
        *,
        alignment_kind,
        alignment_mode_id=None,
        unmatched_speech_ratio,
        mean_local_confidence_speech,
        mean_coarse_confidence_speech,
        context: str,
        require_quality_metrics: bool = True,
    ) -> None:
        normalized_kind = self._normalize_alignment_kind_export(alignment_kind, mode_id=alignment_mode_id)
        if not normalized_kind.startswith("continuous"):
            raise RuntimeError(
                f"minimal_v1 canonical path requires continuous paired-target alignment; got {normalized_kind!r} in {context}."
            )
        if not bool(require_quality_metrics):
            return
        unmatched_max, local_min, coarse_min = self._alignment_quality_thresholds()
        unmatched = float(unmatched_speech_ratio)
        mean_local = float(mean_local_confidence_speech)
        mean_coarse = float(mean_coarse_confidence_speech)
        if unmatched > unmatched_max:
            raise RuntimeError(
                f"minimal_v1 paired target rejected in {context}: unmatched_speech_ratio={unmatched:.3f} > {unmatched_max:.3f}"
            )
        if mean_local < local_min:
            raise RuntimeError(
                f"minimal_v1 paired target rejected in {context}: mean_local_confidence_speech={mean_local:.3f} < {local_min:.3f}"
            )
        if mean_coarse < coarse_min:
            raise RuntimeError(
                f"minimal_v1 paired target rejected in {context}: mean_coarse_confidence_speech={mean_coarse:.3f} < {coarse_min:.3f}"
            )

    def _validate_cached_minimal_continuous_target(self, sample: dict, *, context: str) -> None:
        if "unit_duration_proj_raw_tgt" not in sample:
            raise RuntimeError(
                "minimal_v1_profile + continuous alignment requires explicit unit_duration_proj_raw_tgt "
                f"when loading cached unit_duration_tgt ({context})."
            )
        if "unit_alignment_mode_id_tgt" not in sample:
            raise RuntimeError(
                "minimal_v1_profile + continuous alignment requires unit_alignment_mode_id_tgt "
                f"for cached paired-target supervision ({context})."
            )
        if "unit_alignment_source_tgt" not in sample and "alignment_source" not in sample:
            raise RuntimeError(
                "minimal_v1_profile + continuous alignment requires unit_alignment_source_tgt "
                f"for cached paired-target supervision ({context})."
            )
        if "unit_alignment_version_tgt" not in sample and "alignment_version" not in sample:
            raise RuntimeError(
                "minimal_v1_profile + continuous alignment requires unit_alignment_version_tgt "
                f"for cached paired-target supervision ({context})."
            )
        for key in _MINIMAL_CONTINUOUS_QUALITY_KEYS + _MINIMAL_CONTINUOUS_SIGNATURE_KEYS:
            if key not in sample:
                raise RuntimeError(
                    "minimal_v1_profile + continuous alignment requires "
                    f"{key} for cached paired-target supervision ({context})."
                )
        alignment_mode_id = sample.get("unit_alignment_mode_id_tgt")
        alignment_kind = sample.get("unit_alignment_kind_tgt")
        unmatched = self._extract_optional_float_scalar(sample.get("unit_alignment_unmatched_speech_ratio_tgt"))
        mean_local = self._extract_optional_float_scalar(sample.get("unit_alignment_mean_local_confidence_speech_tgt"))
        mean_coarse = self._extract_optional_float_scalar(sample.get("unit_alignment_mean_coarse_confidence_speech_tgt"))
        self._enforce_minimal_continuous_alignment_gate(
            alignment_kind=alignment_kind,
            alignment_mode_id=alignment_mode_id,
            unmatched_speech_ratio=unmatched or 0.0,
            mean_local_confidence_speech=mean_local or 0.0,
            mean_coarse_confidence_speech=mean_coarse or 0.0,
            context=context,
            require_quality_metrics=True,
        )
        if not self._is_continuous_alignment_kind(alignment_kind, mode_id=alignment_mode_id):
            raise RuntimeError(
                f"minimal_v1_profile cached paired-target supervision must carry continuous provenance ({context})."
            )
        for key in _MINIMAL_CONTINUOUS_SIGNATURE_KEYS:
            signature = self._normalize_signature_text(sample.get(key))
            if signature == "missing":
                raise RuntimeError(
                    "minimal_v1_profile cached paired-target supervision must carry non-empty "
                    f"{key} ({context})."
                )

    def _validate_cached_minimal_continuous_signatures(
        self,
        *,
        sample: dict,
        source_cache: dict | None,
        paired_target_conditioning: dict | None,
        context: str,
    ) -> None:
        cached_source_signature = self._normalize_signature_text(sample.get("unit_alignment_source_cache_signature_tgt"))
        cached_target_signature = self._normalize_signature_text(sample.get("unit_alignment_target_cache_signature_tgt"))
        cached_sidecar_signature = self._normalize_signature_text(sample.get("unit_alignment_sidecar_signature_tgt"))
        current_source_signature = duration_v3_cache_meta_signature(source_cache)
        if (
            current_source_signature != "missing"
            and cached_source_signature != current_source_signature
        ):
            raise RuntimeError(
                "minimal_v1 cached paired-target supervision source signature mismatch: "
                f"cached={cached_source_signature}, current_source={current_source_signature} ({context})"
            )
        if not isinstance(paired_target_conditioning, dict):
            return
        conditioning_source_signature = self._normalize_signature_text(
            paired_target_conditioning.get("source_cache_meta_signature")
        )
        if (
            conditioning_source_signature != "missing"
            and cached_source_signature != conditioning_source_signature
        ):
            raise RuntimeError(
                "minimal_v1 cached paired-target supervision source provenance mismatch: "
                f"cached={cached_source_signature}, conditioning_source={conditioning_source_signature} ({context})"
            )
        conditioning_target_signature = self._normalize_signature_text(
            paired_target_conditioning.get("paired_target_cache_meta_signature")
        )
        if (
            conditioning_target_signature != "missing"
            and cached_target_signature != conditioning_target_signature
        ):
            raise RuntimeError(
                "minimal_v1 cached paired-target supervision target signature mismatch: "
                f"cached={cached_target_signature}, conditioning_target={conditioning_target_signature} ({context})"
            )
        conditioning_sidecar_signature = self._normalize_signature_text(
            paired_target_conditioning.get("paired_target_alignment_sidecar_signature")
        )
        if (
            conditioning_sidecar_signature != "missing"
            and cached_sidecar_signature != conditioning_sidecar_signature
        ):
            raise RuntimeError(
                "minimal_v1 cached paired-target supervision sidecar signature mismatch: "
                f"cached={cached_sidecar_signature}, conditioning_sidecar={conditioning_sidecar_signature} ({context})"
            )

    def _validate_live_minimal_continuous_target_exports(
        self,
        *,
        out: dict,
        context: str,
    ) -> None:
        alignment_kind = self._normalize_alignment_kind_export(
            out.get("unit_alignment_kind_tgt"),
            mode_id=out.get("unit_alignment_mode_id_tgt"),
        )
        if not alignment_kind.startswith("continuous"):
            return
        for key in (
            "unit_alignment_source_tgt",
            "unit_alignment_version_tgt",
            "unit_alignment_source_cache_signature_tgt",
            "unit_alignment_target_cache_signature_tgt",
            "unit_alignment_sidecar_signature_tgt",
        ):
            value = out.get(key)
            text = (
                self._normalize_signature_text(value)
                if key.endswith("_signature_tgt")
                else str(_extract_object_scalar(value) or "").strip() or "missing"
            )
            if text == "missing":
                raise RuntimeError(
                    "minimal_v1 canonical paired-target projection requires non-empty "
                    f"{key} ({context})."
                )
        if alignment_kind == _ALIGNMENT_KIND_CONTINUOUS_PRECOMPUTED:
            source_text = str(_extract_object_scalar(out.get("unit_alignment_source_tgt")) or "").strip()
            version_text = str(_extract_object_scalar(out.get("unit_alignment_version_tgt")) or "").strip()
            if not source_text or not version_text:
                raise RuntimeError(
                    "minimal_v1 canonical paired-target projection forbids metadata-free continuous_precomputed alignment "
                    f"({context})."
                )

    def _allow_prompt_weight_shape_repair(self) -> bool:
        return bool(self.hparams.get(_PROMPT_WEIGHT_REPAIR_FLAG, False))

    def _resolve_duration_v3_unit_prior_path(self) -> str | None:
        path = self.hparams.get("rhythm_v3_unit_prior_path")
        if path is None:
            return None
        text = str(path).strip()
        return text or None

    def _maybe_attach_duration_v3_unit_prior(self, source_cache: dict) -> dict:
        unit_prior_path = self._resolve_duration_v3_unit_prior_path()
        if unit_prior_path is None:
            return source_cache
        if source_cache.get("unit_log_prior") is not None or source_cache.get("prompt_unit_log_prior") is not None:
            return source_cache
        return maybe_attach_unit_log_prior_from_path(
            source_cache,
            unit_prior_path=unit_prior_path,
        )

    @staticmethod
    def _build_prompt_global_weight(
        *,
        prompt_speech_mask: np.ndarray,
        run_stability,
        allow_shape_repair: bool = False,
    ) -> np.ndarray:
        speech = np.asarray(prompt_speech_mask, dtype=np.float32)
        stability = (
            np.ones_like(speech, dtype=np.float32)
            if run_stability is None
            else np.asarray(run_stability, dtype=np.float32)
        )
        if stability.shape != speech.shape:
            if not bool(allow_shape_repair):
                raise RuntimeError(
                    "prompt_run_stability shape mismatch: "
                    f"{tuple(stability.shape)} vs {tuple(speech.shape)}. "
                    f"Set {_PROMPT_WEIGHT_REPAIR_FLAG}=true only for explicit debug repair."
                )
            resized = np.ones_like(speech, dtype=np.float32)
            limit = min(int(resized.size), int(stability.reshape(-1).shape[0]))
            resized.reshape(-1)[:limit] = stability.reshape(-1)[:limit]
            stability = resized
        stability = stability.clip(0.0, 1.0)
        return (speech * (0.25 + (0.75 * stability))).astype(np.float32)

    def _attach_prompt_global_stats_sidecars(self, *, conditioning: dict, source_cache: dict) -> None:
        prompt_duration_obs = np.asarray(conditioning["prompt_duration_obs"], dtype=np.float32)
        prompt_speech_mask = np.asarray(conditioning["prompt_speech_mask"], dtype=np.float32)
        conditioning["prompt_global_weight"] = self._build_prompt_global_weight(
            prompt_speech_mask=prompt_speech_mask,
            run_stability=source_cache.get("source_run_stability"),
            allow_shape_repair=self._allow_prompt_weight_shape_repair(),
        )
        conditioning["prompt_global_weight_present"] = np.asarray([1.0], dtype=np.float32)
        conditioning["g_trim_ratio"] = np.asarray(
            [float(self.hparams.get("rhythm_v3_g_trim_ratio", 0.2) or 0.2)],
            dtype=np.float32,
        )
        g_variant = normalize_global_rate_variant(self.hparams.get("rhythm_v3_g_variant", "raw_median"))
        prompt_unit_log_prior = source_cache.get("prompt_unit_log_prior")
        if prompt_unit_log_prior is None:
            prompt_unit_log_prior = source_cache.get("unit_log_prior")
        unit_prior_meta = source_cache.get(UNIT_LOG_PRIOR_META_KEY, {})
        if prompt_unit_log_prior is not None:
            prompt_unit_log_prior = np.asarray(prompt_unit_log_prior, dtype=np.float32).reshape(-1)
            if prompt_unit_log_prior.shape != prompt_duration_obs.shape:
                raise RuntimeError(
                    "prompt_unit_log_prior must match prompt run shape for maintained prompt conditioning: "
                    f"{prompt_unit_log_prior.shape} vs {prompt_duration_obs.shape}"
                )
            conditioning["prompt_unit_log_prior"] = prompt_unit_log_prior
            conditioning["prompt_unit_log_prior_present"] = np.asarray([1.0], dtype=np.float32)
            conditioning["prompt_unit_prior_vocab_size"] = np.asarray(
                [int(unit_prior_meta.get("unit_prior_vocab_size", 0) or 0)],
                dtype=np.int64,
            )
        elif g_variant == "unit_norm":
            raise RuntimeError(
                "rhythm_v3 g_variant=unit_norm requires prompt_unit_log_prior/unit_log_prior "
                "matching prompt runs in prompt/reference conditioning."
            )
        else:
            conditioning["prompt_unit_log_prior_present"] = np.asarray([0.0], dtype=np.float32)
            conditioning["prompt_unit_prior_vocab_size"] = np.asarray([0], dtype=np.int64)
        if conditioning["prompt_global_weight"].shape != prompt_duration_obs.shape:
            raise RuntimeError(
                "prompt_global_weight shape mismatch with prompt_duration_obs: "
                f"{conditioning['prompt_global_weight'].shape} vs {prompt_duration_obs.shape}"
            )

    def _use_duration_v3_simple_global_stats(self) -> bool:
        if not self._use_duration_v3_dataset_contract():
            return False
        rate_mode = str(self.hparams.get("rhythm_v3_rate_mode", "") or "").strip().lower()
        if rate_mode == "simple_global":
            return True
        return self._is_enabled_flag(self.hparams.get("rhythm_v3_simple_global_stats", False))

    def _use_duration_v3_dataset_contract(self) -> bool:
        return bool(
            self.hparams.get("rhythm_enable_v3", False)
            or is_duration_operator_mode(self.hparams.get("rhythm_mode", ""))
        )

    def _should_emit_duration_v3_silence_runs(self) -> bool:
        if not self._use_duration_v3_dataset_contract():
            return False
        return bool(self.hparams.get("rhythm_v3_emit_silence_runs", True))

    def _should_perturb_duration_v3_source_context(self) -> bool:
        if not self._use_duration_v3_dataset_contract():
            return False
        if str(self.prefix).lower() != "train":
            return False
        if not is_duration_v3_prompt_summary_backbone(self.hparams.get("rhythm_v3_backbone", "global_only")):
            return False
        if str(self.hparams.get("rhythm_v3_anchor_mode", "baseline") or "baseline").strip().lower() != "source_observed":
            return False
        return self._is_enabled_flag(
            self.hparams.get(
                "pseudo_source_duration_perturbation",
                self.hparams.get("rhythm_pseudo_source_duration_perturbation", False),
            )
        )

    def _maybe_build_duration_v3_training_source_cache(self, source_cache: dict, *, item_name: str) -> dict:
        if not self._should_perturb_duration_v3_source_context():
            return source_cache
        source_duration = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32).reshape(1, -1)
        unit_mask = (source_duration > 0.0).astype(np.float32)
        sep_hint = np.asarray(
            source_cache.get("sep_hint", np.zeros_like(source_duration, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1, -1)
        source_silence = np.asarray(
            source_cache.get("source_silence_mask", np.zeros_like(source_duration, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(1, -1)
        generator = self._make_item_generator(item_name=item_name, salt="pseudo_source_duration")
        perturbed = build_pseudo_source_duration_context(
            torch.from_numpy(source_duration),
            torch.from_numpy(unit_mask),
            torch.from_numpy(sep_hint),
            silence_mask=torch.from_numpy(source_silence),
            global_scale_range=(
                float(self.hparams.get("rhythm_pseudo_source_global_scale_min", 0.85)),
                float(self.hparams.get("rhythm_pseudo_source_global_scale_max", 1.15)),
            ),
            local_span_prob=float(self.hparams.get("rhythm_pseudo_source_local_span_prob", 0.20)),
            local_span_scale=(
                float(self.hparams.get("rhythm_pseudo_source_local_scale_min", 0.70)),
                float(self.hparams.get("rhythm_pseudo_source_local_scale_max", 1.30)),
            ),
            mask_prob=float(self.hparams.get("rhythm_pseudo_source_mask_prob", 0.10)),
            flatten_boundary_prob=float(self.hparams.get("rhythm_pseudo_source_flatten_boundary_prob", 0.15)),
            generator=generator,
        ).squeeze(0).cpu().numpy().astype(np.float32)
        runtime_cache = self._copy_numpy_fields(source_cache)
        runtime_cache["dur_anchor_src"] = perturbed
        return runtime_cache

    def _maybe_augment_prompt_unit_conditioning(self, conditioning: dict, *, item_name: str) -> dict:
        if str(self.prefix).lower() != "train":
            return conditioning
        dropout = float(self.hparams.get("rhythm_prompt_dropout", 0.0) or 0.0)
        truncation = float(self.hparams.get("rhythm_prompt_truncation", 0.0) or 0.0)
        if dropout <= 0.0 and truncation <= 0.0:
            return conditioning
        prompt_valid_mask = np.asarray(conditioning.get("prompt_valid_mask", conditioning.get("prompt_unit_mask")), dtype=np.float32).copy()
        prompt_speech_mask = np.asarray(conditioning.get("prompt_speech_mask", prompt_valid_mask), dtype=np.float32).copy()
        if prompt_valid_mask.size <= 0:
            return conditioning
        valid = torch.from_numpy(prompt_valid_mask).reshape(1, -1) > 0.5
        if not bool(valid.any().item()):
            return conditioning
        keep = valid.clone()
        generator = self._make_item_generator(item_name=item_name, salt="prompt_duration_memory")
        if truncation > 0.0:
            valid_indices = torch.nonzero(valid[0], as_tuple=False).reshape(-1)
            valid_count = int(valid_indices.numel())
            if valid_count > 0:
                if truncation < 1.0:
                    min_keep = max(1, int(np.ceil(valid_count * truncation)))
                    max_keep = valid_count
                    if min_keep >= max_keep:
                        keep_units = max_keep
                    else:
                        sampled = torch.randint(min_keep, max_keep + 1, (1,), generator=generator, device=valid.device)
                        keep_units = int(sampled.item())
                else:
                    keep_units = max(1, min(valid_count, int(round(truncation))))
                cutoff_index = int(valid_indices[keep_units - 1].item())
                keep[:, cutoff_index + 1 :] = False
        if dropout > 0.0:
            drop = torch.rand(keep.shape, generator=generator, device=keep.device) < float(max(0.0, min(1.0, dropout)))
            keep = keep & ~drop
        speech_valid = (torch.from_numpy(prompt_speech_mask).reshape(1, -1) > 0.5) & valid
        if not bool((keep & speech_valid).any().item()) and bool(speech_valid.any().item()):
            first_speech = int(torch.nonzero(speech_valid[0], as_tuple=False)[0].item())
            keep[:, first_speech] = True
        if not bool(keep.any().item()):
            first_valid = int(torch.nonzero(valid[0], as_tuple=False)[0].item())
            keep[:, first_valid] = True
        keep_np = keep.float().reshape(-1).cpu().numpy().astype(np.float32)
        augmented = self._copy_numpy_fields(conditioning)
        augmented["prompt_unit_mask"] = keep_np
        augmented["prompt_valid_mask"] = keep_np
        augmented["prompt_speech_mask"] = prompt_speech_mask * keep_np
        for key in (
            "prompt_duration_obs",
            "prompt_unit_anchor_base",
            "prompt_log_base",
            "prompt_global_weight",
            "prompt_unit_log_prior",
            "prompt_source_boundary_cue",
            "prompt_phrase_group_pos",
            "prompt_phrase_final_mask",
        ):
            if key in augmented:
                augmented[key] = np.asarray(augmented[key], dtype=np.float32) * keep_np
        return augmented

    def _build_reference_prompt_unit_conditioning(self, prompt_item, *, target_mode: str):
        if prompt_item is None:
            return {}
        source_cache = None
        item_name = str(prompt_item.get("item_name", "<prompt-item>")) if isinstance(prompt_item, dict) else "<prompt-item>"
        explicit_silence = self._should_emit_duration_v3_silence_runs()
        has_cached_prompt_source = all(key in prompt_item for key in ("content_units", "dur_anchor_src"))
        has_prompt_silence = "source_silence_mask" in prompt_item
        if has_cached_prompt_source and (not explicit_silence or has_prompt_silence):
            source_cache = {
                "content_units": prompt_item["content_units"],
                "dur_anchor_src": prompt_item["dur_anchor_src"],
            }
            for extra_key in ("unit_anchor_base", "unit_rate_log_base"):
                if extra_key in prompt_item:
                    source_cache[extra_key] = prompt_item[extra_key]
            if has_prompt_silence:
                source_cache["source_silence_mask"] = prompt_item["source_silence_mask"]
            if "sep_hint" in prompt_item:
                source_cache["sep_hint"] = prompt_item["sep_hint"]
            for extra_key in ("source_boundary_cue", "phrase_group_pos", "phrase_final_mask", "source_run_stability", "unit_log_prior"):
                if extra_key in prompt_item:
                    source_cache[extra_key] = prompt_item[extra_key]
        elif target_mode != "cached_only" and "hubert" in prompt_item:
            source_cache = build_source_rhythm_cache(
                np.asarray(prompt_item["hubert"]),
                silent_token=self.hparams.get("silent_token", 57),
                separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                emit_silence_runs=explicit_silence,
                debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
                unit_prior_path=self._resolve_duration_v3_unit_prior_path(),
            )
        elif has_cached_prompt_source and target_mode != "cached_only":
            if explicit_silence:
                if "hubert" in prompt_item:
                    source_cache = build_source_rhythm_cache(
                        np.asarray(prompt_item["hubert"]),
                        silent_token=self.hparams.get("silent_token", 57),
                        separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                        tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                        emit_silence_runs=True,
                        debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                        phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
                        unit_prior_path=self._resolve_duration_v3_unit_prior_path(),
                    )
                else:
                    raise RuntimeError(
                        "explicit silence-run frontend requires source_silence_mask in prompt/reference caches. "
                        "Rebuild the cache or provide raw hubert."
                    )
            else:
                source_cache = {
                    "content_units": prompt_item["content_units"],
                    "dur_anchor_src": prompt_item["dur_anchor_src"],
                }
                for extra_key in ("unit_anchor_base", "unit_rate_log_base"):
                    if extra_key in prompt_item:
                        source_cache[extra_key] = prompt_item[extra_key]
                if "sep_hint" in prompt_item:
                    source_cache["sep_hint"] = prompt_item["sep_hint"]
                for extra_key in ("source_boundary_cue", "phrase_group_pos", "phrase_final_mask", "source_run_stability", "unit_log_prior"):
                    if extra_key in prompt_item:
                        source_cache[extra_key] = prompt_item[extra_key]
        elif target_mode == "cached_only" and explicit_silence and has_cached_prompt_source and not has_prompt_silence:
            raise RuntimeError(
                "rhythm_v3 cached_only with explicit silence-run frontend requires source_silence_mask in prompt/reference caches. "
                "Re-binarize with explicit silence runs enabled."
            )
        if source_cache is None:
            return {}
        source_cache = self._maybe_attach_duration_v3_unit_prior(source_cache)
        prompt_content_units = np.asarray(source_cache["content_units"], dtype=np.int64)
        prompt_duration_obs = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
        prompt_valid_mask = (prompt_duration_obs > 0).astype(np.float32)
        prompt_silence_mask = _resolve_run_silence_mask(
            size=prompt_duration_obs.shape[0],
            silence_mask=source_cache.get("source_silence_mask"),
        )
        prompt_speech_mask = prompt_valid_mask * (1.0 - prompt_silence_mask.clip(0.0, 1.0))
        conditioning = {
            "prompt_content_units": prompt_content_units,
            "prompt_duration_obs": prompt_duration_obs,
            "prompt_unit_mask": prompt_valid_mask,
            "prompt_valid_mask": prompt_valid_mask,
            "prompt_speech_mask": prompt_speech_mask,
            "prompt_source_boundary_cue": np.asarray(
                source_cache.get("source_boundary_cue", np.zeros_like(prompt_duration_obs)),
                dtype=np.float32,
            ),
            "prompt_phrase_group_pos": np.asarray(
                source_cache.get("phrase_group_pos", np.zeros_like(prompt_duration_obs)),
                dtype=np.float32,
            ),
            "prompt_phrase_final_mask": np.asarray(
                source_cache.get("phrase_final_mask", np.zeros_like(prompt_duration_obs)),
                dtype=np.float32,
            ),
        }
        use_log_base_rate = not self._use_duration_v3_simple_global_stats()
        if use_log_base_rate and "unit_anchor_base" in source_cache:
            conditioning["prompt_unit_anchor_base"] = np.asarray(source_cache["unit_anchor_base"], dtype=np.float32)
        if use_log_base_rate and "unit_rate_log_base" in source_cache:
            conditioning["prompt_log_base"] = np.asarray(source_cache["unit_rate_log_base"], dtype=np.float32)
        self._attach_prompt_global_stats_sidecars(
            conditioning=conditioning,
            source_cache=source_cache,
        )
        return self._maybe_augment_prompt_unit_conditioning(
            conditioning,
            item_name=item_name,
        )

    @staticmethod
    def _resolve_paired_target_projection_inputs(
        paired_target_conditioning: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if not isinstance(paired_target_conditioning, dict):
            return None
        target_units = paired_target_conditioning.get("paired_target_content_units")
        target_duration = paired_target_conditioning.get("paired_target_duration_obs")
        target_valid = paired_target_conditioning.get("paired_target_valid_mask")
        target_speech = paired_target_conditioning.get("paired_target_speech_mask")
        if any(value is None for value in (target_units, target_duration, target_valid, target_speech)):
            return None
        return (
            _as_int64_1d(target_units),
            _as_float32_1d(target_duration),
            _as_float32_1d(target_valid),
            _as_float32_1d(target_speech),
        )

    @staticmethod
    def _resolve_paired_target_alignment_metadata(
        paired_target_conditioning: dict,
    ) -> dict[str, np.ndarray] | None:
        if not isinstance(paired_target_conditioning, dict):
            return None
        assigned_source = paired_target_conditioning.get("paired_target_alignment_assigned_source")
        if assigned_source is None:
            assigned_source = paired_target_conditioning.get("paired_target_assigned_source")
        assigned_cost = paired_target_conditioning.get("paired_target_alignment_assigned_cost")
        if assigned_cost is None:
            assigned_cost = paired_target_conditioning.get("paired_target_assigned_cost")
        if assigned_source is None and assigned_cost is None:
            return None
        alignment_kind = paired_target_conditioning.get("paired_target_alignment_kind")
        if alignment_kind is None:
            alignment_kind = paired_target_conditioning.get("paired_target_alignment_mode")
        if alignment_kind is None:
            alignment_kind = paired_target_conditioning.get("unit_alignment_kind_tgt")
        alignment_mode_id = paired_target_conditioning.get("paired_target_alignment_mode_id")
        if alignment_mode_id is None:
            alignment_mode_id = paired_target_conditioning.get("paired_target_alignment_kind_id")
        if alignment_mode_id is None:
            alignment_mode_id = paired_target_conditioning.get("unit_alignment_mode_id_tgt")
        alignment_source = paired_target_conditioning.get("paired_target_alignment_source")
        if alignment_source is None:
            alignment_source = paired_target_conditioning.get("alignment_source")
        if alignment_source is None:
            alignment_source = paired_target_conditioning.get("unit_alignment_source_tgt")
        alignment_version = paired_target_conditioning.get("paired_target_alignment_version")
        if alignment_version is None:
            alignment_version = paired_target_conditioning.get("alignment_version")
        if alignment_version is None:
            alignment_version = paired_target_conditioning.get("unit_alignment_version_tgt")
        try:
            normalized_mode_id = int(_extract_object_scalar(alignment_mode_id)) if alignment_mode_id is not None else None
        except Exception:
            normalized_mode_id = None
        resolved_kind = DurationV3DatasetMixin._normalize_alignment_kind_export(
            alignment_kind,
            mode_id=normalized_mode_id,
        )
        if not resolved_kind.startswith("continuous"):
            return None
        normalized_source = str(_extract_object_scalar(alignment_source) or "").strip()
        normalized_version = str(_extract_object_scalar(alignment_version) or "").strip()
        if not normalized_source or not normalized_version:
            return None
        return {
            "assigned_source": _as_int64_1d(assigned_source) if assigned_source is not None else None,
            "assigned_cost": _as_float32_1d(assigned_cost) if assigned_cost is not None else None,
            "alignment_kind": resolved_kind,
            "alignment_source": normalized_source,
            "alignment_version": normalized_version,
        }

    @staticmethod
    def _build_alignment_provenance_exports(
        *,
        alignment_kind: str,
        projection: dict | None = None,
        paired_target_conditioning: dict | None = None,
    ) -> dict[str, np.ndarray]:
        projection = projection if isinstance(projection, dict) else {}
        paired_target_conditioning = (
            paired_target_conditioning if isinstance(paired_target_conditioning, dict) else {}
        )
        defaults = (
            _ALIGNMENT_PROVENANCE_VITERBI_DEFAULTS
            if alignment_kind == _ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1
            else {}
        )
        float_aliases = {
            "unit_alignment_band_ratio_tgt": (
                "alignment_band_ratio",
                "paired_target_alignment_band_ratio",
                "unit_alignment_band_ratio_tgt",
            ),
            "unit_alignment_lambda_emb_tgt": (
                "alignment_lambda_emb",
                "paired_target_alignment_lambda_emb",
                "unit_alignment_lambda_emb_tgt",
            ),
            "unit_alignment_lambda_type_tgt": (
                "alignment_lambda_type",
                "paired_target_alignment_lambda_type",
                "unit_alignment_lambda_type_tgt",
            ),
            "unit_alignment_lambda_band_tgt": (
                "alignment_lambda_band",
                "paired_target_alignment_lambda_band",
                "unit_alignment_lambda_band_tgt",
            ),
            "unit_alignment_lambda_unit_tgt": (
                "alignment_lambda_unit",
                "paired_target_alignment_lambda_unit",
                "unit_alignment_lambda_unit_tgt",
            ),
            "unit_alignment_bad_cost_threshold_tgt": (
                "alignment_bad_cost_threshold",
                "paired_target_alignment_bad_cost_threshold",
                "unit_alignment_bad_cost_threshold_tgt",
            ),
            "unit_alignment_allow_source_skip_tgt": (
                "alignment_allow_source_skip",
                "paired_target_alignment_allow_source_skip",
                "unit_alignment_allow_source_skip_tgt",
            ),
            "unit_alignment_skip_penalty_tgt": (
                "alignment_skip_penalty",
                "paired_target_alignment_skip_penalty",
                "unit_alignment_skip_penalty_tgt",
            ),
        }
        exports: dict[str, np.ndarray] = {}
        for export_key, aliases in float_aliases.items():
            value = None
            for alias in aliases:
                value = DurationV3DatasetMixin._extract_optional_float_scalar(projection.get(alias))
                if value is not None:
                    break
                value = DurationV3DatasetMixin._extract_optional_float_scalar(
                    paired_target_conditioning.get(alias)
                )
                if value is not None:
                    break
            if value is None:
                value = defaults.get(export_key, np.nan)
            exports[export_key] = np.asarray([float(value)], dtype=np.float32)
        for export_key, aliases in {
            "unit_alignment_source_cache_signature_tgt": (
                "source_cache_meta_signature",
                "paired_target_alignment_source_cache_signature",
                "unit_alignment_source_cache_signature_tgt",
            ),
            "unit_alignment_target_cache_signature_tgt": (
                "paired_target_cache_meta_signature",
                "paired_target_alignment_target_cache_signature",
                "unit_alignment_target_cache_signature_tgt",
            ),
            "unit_alignment_sidecar_signature_tgt": (
                "paired_target_alignment_sidecar_signature",
                "unit_alignment_sidecar_signature_tgt",
            ),
        }.items():
            resolved = ""
            for alias in aliases:
                resolved = str(_extract_object_scalar(projection.get(alias)) or "").strip()
                if resolved:
                    break
                resolved = str(_extract_object_scalar(paired_target_conditioning.get(alias)) or "").strip()
                if resolved:
                    break
            exports[export_key] = np.asarray([resolved], dtype=object)
        return exports

    def _alignment_provenance_optional_keys(self) -> tuple[str, ...]:
        return _ALIGNMENT_PROVENANCE_FLOAT_KEYS + _ALIGNMENT_PROVENANCE_OBJECT_KEYS + (
            "unit_alignment_is_continuous_tgt",
            "unit_alignment_coverage_binary_tgt",
            "unit_alignment_coverage_fraction_tgt",
            "unit_alignment_expected_frame_support_tgt",
            "unit_alignment_confidence_cost_term_tgt",
            "unit_alignment_confidence_margin_term_tgt",
            "unit_alignment_confidence_type_term_tgt",
            "unit_alignment_confidence_match_term_tgt",
        )

    def _resolve_optional_sample_keys(self) -> tuple[str, ...]:
        keys = list(super()._resolve_optional_sample_keys())
        keys.extend(self._alignment_provenance_optional_keys())
        return tuple(dict.fromkeys(keys))

    def _build_optional_collate_spec(self) -> dict[str, tuple[str, float | int | None]]:
        spec = dict(super()._build_optional_collate_spec())
        for key in _ALIGNMENT_PROVENANCE_FLOAT_KEYS:
            spec[key] = ("float", 0.0)
        for key in _ALIGNMENT_PROVENANCE_OBJECT_KEYS:
            spec[key] = ("object", None)
        for key in (
            "unit_alignment_is_continuous_tgt",
            "unit_alignment_coverage_binary_tgt",
            "unit_alignment_coverage_fraction_tgt",
            "unit_alignment_expected_frame_support_tgt",
            "unit_alignment_confidence_cost_term_tgt",
            "unit_alignment_confidence_margin_term_tgt",
            "unit_alignment_confidence_type_term_tgt",
            "unit_alignment_confidence_match_term_tgt",
        ):
            spec[key] = ("float", 0.0)
        return spec

    @staticmethod
    def _maybe_extract_frame_sidecar(item, *keys: str, dtype=None):
        if not isinstance(item, dict):
            return None
        for key in keys:
            if key in item and item[key] is not None:
                return np.asarray(item[key], dtype=dtype) if dtype is not None else np.asarray(item[key])
        return None

    def _build_paired_target_projection_conditioning(self, paired_target_item, *, target_mode: str, source_item=None):
        if paired_target_item is None:
            return {}
        source_cache = None
        item_name = (
            str(paired_target_item.get("item_name", "<paired-target-item>"))
            if isinstance(paired_target_item, dict)
            else "<paired-target-item>"
        )
        explicit_silence = self._should_emit_duration_v3_silence_runs()
        has_cached_target_source = (
            isinstance(paired_target_item, dict)
            and all(key in paired_target_item for key in ("content_units", "dur_anchor_src"))
        )
        has_target_silence = isinstance(paired_target_item, dict) and "source_silence_mask" in paired_target_item
        if has_cached_target_source and (not explicit_silence or has_target_silence):
            source_cache = {
                "content_units": paired_target_item["content_units"],
                "dur_anchor_src": paired_target_item["dur_anchor_src"],
            }
            if has_target_silence:
                source_cache["source_silence_mask"] = paired_target_item["source_silence_mask"]
        elif target_mode != "cached_only" and isinstance(paired_target_item, dict) and "hubert" in paired_target_item:
            source_cache = build_source_rhythm_cache(
                np.asarray(paired_target_item["hubert"]),
                silent_token=self.hparams.get("silent_token", 57),
                separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                emit_silence_runs=explicit_silence,
                debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
                unit_prior_path=self._resolve_duration_v3_unit_prior_path(),
            )
        elif has_cached_target_source and target_mode != "cached_only":
            if explicit_silence:
                if isinstance(paired_target_item, dict) and "hubert" in paired_target_item:
                    source_cache = build_source_rhythm_cache(
                        np.asarray(paired_target_item["hubert"]),
                        silent_token=self.hparams.get("silent_token", 57),
                        separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                        tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                        emit_silence_runs=True,
                        debounce_min_run_frames=int(self.hparams.get("rhythm_v3_debounce_min_run_frames", 2)),
                        phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
                        unit_prior_path=self._resolve_duration_v3_unit_prior_path(),
                    )
                else:
                    raise RuntimeError(
                        "explicit silence-run frontend requires source_silence_mask in cached paired-target sources. "
                        "Rebuild the cache or provide raw hubert."
                    )
            else:
                source_cache = {
                    "content_units": paired_target_item["content_units"],
                    "dur_anchor_src": paired_target_item["dur_anchor_src"],
                }
        elif target_mode == "cached_only" and explicit_silence and has_cached_target_source and not has_target_silence:
            raise RuntimeError(
                "rhythm_v3 cached_only paired-target supervision with explicit silence-run frontend requires "
                "source_silence_mask in paired-target caches. Re-binarize with explicit silence runs enabled."
            )
        if source_cache is None:
            return {}
        paired_target_content_units = np.asarray(source_cache["content_units"], dtype=np.int64)
        paired_target_duration_obs = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
        paired_target_valid_mask = (paired_target_duration_obs > 0).astype(np.float32)
        paired_target_silence_mask = _resolve_run_silence_mask(
            size=paired_target_duration_obs.shape[0],
            silence_mask=source_cache.get("source_silence_mask"),
        )
        paired_target_speech_mask = paired_target_valid_mask * (1.0 - paired_target_silence_mask.clip(0.0, 1.0))
        conditioning = {
            "paired_target_content_units": paired_target_content_units,
            "paired_target_duration_obs": paired_target_duration_obs,
            "paired_target_valid_mask": paired_target_valid_mask,
            "paired_target_speech_mask": paired_target_speech_mask,
            "paired_target_item_name": np.asarray([item_name], dtype=object),
            "paired_target_text_signature": np.asarray(
                [self._rhythm_text_signature(paired_target_item if isinstance(paired_target_item, dict) else None)],
                dtype=object,
            ),
            "source_text_signature": np.asarray(
                [self._rhythm_text_signature(source_item if isinstance(source_item, dict) else None)],
                dtype=object,
            ),
        }
        for src_key, dst_key in (
            ("paired_target_alignment_kind", "paired_target_alignment_kind"),
            ("unit_alignment_kind_tgt", "paired_target_alignment_kind"),
            ("paired_target_alignment_mode", "paired_target_alignment_mode"),
            ("paired_target_alignment_mode_id", "paired_target_alignment_mode_id"),
            ("paired_target_alignment_kind_id", "paired_target_alignment_kind_id"),
            ("unit_alignment_mode_id_tgt", "paired_target_alignment_mode_id"),
            ("paired_target_alignment_source", "paired_target_alignment_source"),
            ("alignment_source", "paired_target_alignment_source"),
            ("unit_alignment_source_tgt", "paired_target_alignment_source"),
            ("paired_target_alignment_version", "paired_target_alignment_version"),
            ("alignment_version", "paired_target_alignment_version"),
            ("unit_alignment_version_tgt", "paired_target_alignment_version"),
            ("paired_target_alignment_band_ratio", "paired_target_alignment_band_ratio"),
            ("alignment_band_ratio", "paired_target_alignment_band_ratio"),
            ("unit_alignment_band_ratio_tgt", "paired_target_alignment_band_ratio"),
            ("paired_target_alignment_lambda_emb", "paired_target_alignment_lambda_emb"),
            ("alignment_lambda_emb", "paired_target_alignment_lambda_emb"),
            ("unit_alignment_lambda_emb_tgt", "paired_target_alignment_lambda_emb"),
            ("paired_target_alignment_lambda_type", "paired_target_alignment_lambda_type"),
            ("alignment_lambda_type", "paired_target_alignment_lambda_type"),
            ("unit_alignment_lambda_type_tgt", "paired_target_alignment_lambda_type"),
            ("paired_target_alignment_lambda_band", "paired_target_alignment_lambda_band"),
            ("alignment_lambda_band", "paired_target_alignment_lambda_band"),
            ("unit_alignment_lambda_band_tgt", "paired_target_alignment_lambda_band"),
            ("paired_target_alignment_lambda_unit", "paired_target_alignment_lambda_unit"),
            ("alignment_lambda_unit", "paired_target_alignment_lambda_unit"),
            ("unit_alignment_lambda_unit_tgt", "paired_target_alignment_lambda_unit"),
            ("paired_target_alignment_bad_cost_threshold", "paired_target_alignment_bad_cost_threshold"),
            ("alignment_bad_cost_threshold", "paired_target_alignment_bad_cost_threshold"),
            ("unit_alignment_bad_cost_threshold_tgt", "paired_target_alignment_bad_cost_threshold"),
            ("paired_target_alignment_allow_source_skip", "paired_target_alignment_allow_source_skip"),
            ("alignment_allow_source_skip", "paired_target_alignment_allow_source_skip"),
            ("unit_alignment_allow_source_skip_tgt", "paired_target_alignment_allow_source_skip"),
            ("paired_target_alignment_skip_penalty", "paired_target_alignment_skip_penalty"),
            ("alignment_skip_penalty", "paired_target_alignment_skip_penalty"),
            ("unit_alignment_skip_penalty_tgt", "paired_target_alignment_skip_penalty"),
            ("paired_target_alignment_source_cache_signature", "paired_target_alignment_source_cache_signature"),
            ("unit_alignment_source_cache_signature_tgt", "paired_target_alignment_source_cache_signature"),
            ("paired_target_alignment_target_cache_signature", "paired_target_alignment_target_cache_signature"),
            ("unit_alignment_target_cache_signature_tgt", "paired_target_alignment_target_cache_signature"),
            ("paired_target_alignment_sidecar_signature", "paired_target_alignment_sidecar_signature"),
            ("unit_alignment_sidecar_signature_tgt", "paired_target_alignment_sidecar_signature"),
            ("paired_target_alignment_assigned_source", "paired_target_alignment_assigned_source"),
            ("paired_target_assigned_source", "paired_target_alignment_assigned_source"),
            ("unit_alignment_assigned_source_debug", "paired_target_alignment_assigned_source"),
            ("paired_target_alignment_assigned_cost", "paired_target_alignment_assigned_cost"),
            ("paired_target_assigned_cost", "paired_target_alignment_assigned_cost"),
            ("unit_alignment_assigned_cost_debug", "paired_target_alignment_assigned_cost"),
        ):
            if not isinstance(paired_target_item, dict):
                continue
            if src_key in paired_target_item and paired_target_item[src_key] is not None and dst_key not in conditioning:
                conditioning[dst_key] = paired_target_item[src_key]
        source_frame_states = self._maybe_extract_frame_sidecar(
            source_item,
            "source_frame_states",
            "frame_states",
            dtype=np.float32,
        )
        source_frame_to_run = self._maybe_extract_frame_sidecar(
            source_item,
            "source_frame_to_run",
            "frame_to_run",
            dtype=np.int64,
        )
        target_frame_states = self._maybe_extract_frame_sidecar(
            paired_target_item,
            "paired_target_frame_states",
            "target_frame_states",
            "frame_states",
            dtype=np.float32,
        )
        target_frame_speech_prob = self._maybe_extract_frame_sidecar(
            paired_target_item,
            "paired_target_frame_speech_prob",
            "target_frame_speech_prob",
            "frame_speech_prob",
            "frame_speech_mask",
            dtype=np.float32,
        )
        target_frame_weight = self._maybe_extract_frame_sidecar(
            paired_target_item,
            "paired_target_frame_weight",
            "target_frame_weight",
            "frame_weight",
            dtype=np.float32,
        )
        target_frame_valid = self._maybe_extract_frame_sidecar(
            paired_target_item,
            "paired_target_frame_valid",
            "target_frame_valid",
            "frame_valid",
            dtype=np.float32,
        )
        target_frame_unit_hint = self._maybe_extract_frame_sidecar(
            paired_target_item,
            "paired_target_frame_unit_hint",
            "target_frame_unit_hint",
            "frame_unit_hint",
            dtype=np.int64,
        )
        if source_frame_states is not None:
            conditioning["source_frame_states"] = np.asarray(source_frame_states, dtype=np.float32)
        if source_frame_to_run is not None:
            conditioning["source_frame_to_run"] = np.asarray(source_frame_to_run, dtype=np.int64)
        if target_frame_states is not None:
            conditioning["paired_target_frame_states"] = np.asarray(target_frame_states, dtype=np.float32)
        if target_frame_speech_prob is not None:
            conditioning["paired_target_frame_speech_prob"] = np.asarray(target_frame_speech_prob, dtype=np.float32)
        if target_frame_weight is not None:
            conditioning["paired_target_frame_weight"] = np.asarray(target_frame_weight, dtype=np.float32)
        if target_frame_valid is not None:
            conditioning["paired_target_frame_valid"] = np.asarray(target_frame_valid, dtype=np.float32)
        if target_frame_unit_hint is not None:
            conditioning["paired_target_frame_unit_hint"] = np.asarray(target_frame_unit_hint, dtype=np.int64)
        source_cache_signature = duration_v3_cache_meta_signature(source_item if isinstance(source_item, dict) else None)
        paired_target_cache_signature = duration_v3_cache_meta_signature(
            paired_target_item if isinstance(paired_target_item, dict) else None
        )
        if "source_cache_meta_signature" not in conditioning:
            conditioning["source_cache_meta_signature"] = np.asarray([source_cache_signature], dtype=object)
        if "paired_target_cache_meta_signature" not in conditioning:
            conditioning["paired_target_cache_meta_signature"] = np.asarray(
                [paired_target_cache_signature],
                dtype=object,
            )
        if "paired_target_alignment_source_cache_signature" not in conditioning:
            conditioning["paired_target_alignment_source_cache_signature"] = np.asarray(
                [source_cache_signature],
                dtype=object,
            )
        if "paired_target_alignment_target_cache_signature" not in conditioning:
            conditioning["paired_target_alignment_target_cache_signature"] = np.asarray(
                [paired_target_cache_signature],
                dtype=object,
            )
        if "paired_target_alignment_sidecar_signature" not in conditioning:
            conditioning["paired_target_alignment_sidecar_signature"] = np.asarray(
                [
                    self._build_alignment_sidecar_signature(
                        source_cache_meta_signature=source_cache_signature,
                        target_cache_meta_signature=paired_target_cache_signature,
                        source_frame_states=source_frame_states,
                        source_frame_to_run=source_frame_to_run,
                        target_frame_states=target_frame_states,
                        target_frame_speech_prob=target_frame_speech_prob,
                        target_frame_weight=target_frame_weight,
                        target_frame_valid=target_frame_valid,
                        target_frame_unit_hint=target_frame_unit_hint,
                    )
                ],
                dtype=object,
            )
        return conditioning

    def _build_paired_duration_v3_targets(self, *, item, source_cache, paired_target_conditioning):
        projection_inputs = self._resolve_paired_target_projection_inputs(paired_target_conditioning)
        if projection_inputs is None:
            return None
        item_name = str(item.get("item_name", "<unknown-item>")) if isinstance(item, dict) else "<unknown-item>"
        source_units = _as_int64_1d(source_cache["content_units"])
        source_duration = _as_float32_1d(source_cache["dur_anchor_src"])
        source_silence = _resolve_run_silence_mask(
            size=source_duration.shape[0],
            silence_mask=source_cache.get("source_silence_mask"),
        )
        target_units, target_duration, target_valid, target_speech = projection_inputs
        precomputed_alignment = self._resolve_paired_target_alignment_metadata(
            paired_target_conditioning
        )
        source_frame_states = paired_target_conditioning.get("source_frame_states")
        source_frame_to_run = paired_target_conditioning.get("source_frame_to_run")
        target_frame_states = paired_target_conditioning.get("paired_target_frame_states")
        target_frame_speech_prob = paired_target_conditioning.get("paired_target_frame_speech_prob")
        target_frame_weight = paired_target_conditioning.get("paired_target_frame_weight")
        target_frame_valid = paired_target_conditioning.get("paired_target_frame_valid")
        target_frame_unit_hint = paired_target_conditioning.get("paired_target_frame_unit_hint")
        use_continuous_alignment = bool(
            self.hparams.get("rhythm_v3_use_continuous_alignment", False)
        )
        alignment_soft_repair = self._is_enabled_flag(
            self.hparams.get("rhythm_v3_alignment_soft_repair", False)
        )
        configured_alignment_mode_raw = self.hparams.get("rhythm_v3_alignment_mode", None)
        configured_alignment_mode = (
            str(_extract_object_scalar(configured_alignment_mode_raw) or "").strip().lower()
            if configured_alignment_mode_raw is not None
            else "auto"
        )
        if configured_alignment_mode and configured_alignment_mode not in {
            "auto",
            _ALIGNMENT_KIND_DISCRETE,
            _ALIGNMENT_KIND_CONTINUOUS_PRECOMPUTED,
            _ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1,
        }:
            raise RuntimeError(
                "Unsupported rhythm_v3_alignment_mode. "
                f"Expected one of: auto, {_ALIGNMENT_KIND_DISCRETE}, "
                f"{_ALIGNMENT_KIND_CONTINUOUS_PRECOMPUTED}, {_ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1}; "
                f"got {configured_alignment_mode_raw!r}"
            )
        if use_continuous_alignment and configured_alignment_mode == _ALIGNMENT_KIND_DISCRETE:
            raise RuntimeError(
                "rhythm_v3_use_continuous_alignment=true requires rhythm_v3_alignment_mode "
                "to be continuous_precomputed or continuous_viterbi_v1."
            )
        continuous_aligner_kwargs = None
        if use_continuous_alignment and configured_alignment_mode == _ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1:
            self._validate_continuous_projection_provenance(
                source_cache=source_cache,
                paired_target_conditioning=paired_target_conditioning,
                context=item_name,
            )
            band_width = self._resolve_hparam_alias(
                "rhythm_v3_alignment_band_width",
                "rhythm_v3_align_band_width",
            )
            continuous_aligner_kwargs = {
                "lambda_emb": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_lambda_emb",
                        "rhythm_v3_align_lambda_emb",
                        default=1.0,
                    )
                ),
                "lambda_type": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_lambda_type",
                        "rhythm_v3_align_lambda_type",
                        default=0.5,
                    )
                ),
                "lambda_band": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_lambda_band",
                        "rhythm_v3_align_lambda_band",
                        default=0.2,
                    )
                ),
                "lambda_unit": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_lambda_unit",
                        "rhythm_v3_align_lambda_unit",
                        default=0.0,
                    )
                ),
                "band_ratio": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_band_ratio",
                        "rhythm_v3_align_band_ratio",
                        default=0.08,
                    )
                ),
                "bad_cost_threshold": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_bad_cost_threshold",
                        "rhythm_v3_align_bad_cost_threshold",
                        default=1.2,
                    )
                ),
                "allow_source_skip": self._is_enabled_flag(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_allow_source_skip",
                        "rhythm_v3_align_allow_source_skip",
                        default=False,
                    )
                ),
                "skip_penalty": float(
                    self._resolve_hparam_alias(
                        "rhythm_v3_alignment_skip_penalty",
                        "rhythm_v3_align_skip_penalty",
                        default=1.0,
                    )
                ),
            }
            if band_width is not None:
                continuous_aligner_kwargs["band_width"] = int(band_width)
        allow_source_self_target_fallback = self._is_enabled_flag(
            self.hparams.get("rhythm_v3_allow_source_self_target_fallback", False)
        )
        try:
            projection = _project_target_runs_onto_source(
                source_units=source_units,
                source_durations=source_duration,
                source_silence_mask=source_silence,
                target_units=target_units,
                target_durations=target_duration,
                target_valid_mask=target_valid,
                target_speech_mask=target_speech,
                use_continuous_alignment=use_continuous_alignment,
                source_frame_states=source_frame_states,
                target_frame_states=target_frame_states,
                source_frame_to_run=source_frame_to_run,
                target_frame_speech_prob=target_frame_speech_prob,
                target_frame_weight=target_frame_weight,
                target_frame_valid=target_frame_valid,
                target_frame_unit_hint=target_frame_unit_hint,
                continuous_alignment_mode=configured_alignment_mode,
                continuous_aligner_kwargs=continuous_aligner_kwargs,
                precomputed_alignment=precomputed_alignment,
                allow_source_self_target_fallback=allow_source_self_target_fallback,
                alignment_soft_repair=alignment_soft_repair,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build paired duration_v3 targets for {item_name}: {exc}") from exc
        projected = np.asarray(projection["projected"], dtype=np.float32)
        confidence_local = np.asarray(projection["confidence_local"], dtype=np.float32)
        confidence_coarse = np.asarray(projection["confidence_coarse"], dtype=np.float32)
        coverage = np.asarray(projection["coverage"], dtype=np.float32)
        coverage_binary = np.asarray(
            projection.get("coverage_binary", coverage),
            dtype=np.float32,
        )
        coverage_fraction = np.asarray(
            projection.get("coverage_fraction", coverage),
            dtype=np.float32,
        )
        expected_frame_support = np.asarray(
            projection.get("expected_frame_support", np.zeros_like(coverage, dtype=np.float32)),
            dtype=np.float32,
        )
        match_rate = np.asarray(projection["match_rate"], dtype=np.float32)
        mean_cost = np.asarray(projection["mean_cost"], dtype=np.float32)
        confidence_cost_term = np.asarray(
            projection.get("confidence_cost_term", np.zeros_like(confidence_local, dtype=np.float32)),
            dtype=np.float32,
        )
        confidence_margin_term = np.asarray(
            projection.get("confidence_margin_term", np.zeros_like(confidence_local, dtype=np.float32)),
            dtype=np.float32,
        )
        confidence_type_term = np.asarray(
            projection.get("confidence_type_term", np.zeros_like(confidence_local, dtype=np.float32)),
            dtype=np.float32,
        )
        confidence_match_term = np.asarray(
            projection.get("confidence_match_term", np.zeros_like(confidence_local, dtype=np.float32)),
            dtype=np.float32,
        )
        if (
            projected.shape[0] != source_duration.shape[0]
            or confidence_local.shape[0] != source_duration.shape[0]
            or confidence_coarse.shape[0] != source_duration.shape[0]
        ):
            raise RuntimeError(
                f"Paired duration_v3 projection length mismatch for {item_name}: "
                f"source={source_duration.shape[0]}, projected={projected.shape[0]}, "
                f"confidence_local={confidence_local.shape[0]}, confidence_coarse={confidence_coarse.shape[0]}"
            )
        alignment_kind = self._normalize_alignment_kind_export(projection.get("alignment_kind", _ALIGNMENT_KIND_DISCRETE))
        alignment_mode_id = 0
        if alignment_kind == _ALIGNMENT_KIND_CONTINUOUS_PRECOMPUTED:
            alignment_mode_id = _ALIGNMENT_MODE_ID_CONTINUOUS_PRECOMPUTED
        elif alignment_kind == _ALIGNMENT_KIND_CONTINUOUS_VITERBI_V1:
            alignment_mode_id = _ALIGNMENT_MODE_ID_CONTINUOUS_VITERBI_V1
        if self._is_enabled_flag(self.hparams.get("rhythm_v3_minimal_v1_profile", False)):
            self._enforce_minimal_continuous_alignment_gate(
                alignment_kind=alignment_kind,
                unmatched_speech_ratio=projection.get("unmatched_speech_ratio", 0.0),
                mean_local_confidence_speech=projection.get("mean_local_confidence_speech", 0.0),
                mean_coarse_confidence_speech=projection.get("mean_coarse_confidence_speech", 0.0),
                context=item_name,
            )
        out = {
            "unit_duration_tgt": projected.astype(np.float32),
            "unit_duration_proj_raw_tgt": projected.astype(np.float32),
            "unit_logstretch_proj_raw_tgt": (
                np.log(np.clip(projected.astype(np.float32), 1.0e-6, None))
                - np.log(np.clip(source_duration.astype(np.float32), 1.0e-6, None))
            ).astype(np.float32),
            "unit_confidence_local_tgt": confidence_local.astype(np.float32),
            "unit_confidence_coarse_tgt": confidence_coarse.astype(np.float32),
            "unit_confidence_tgt": confidence_coarse.astype(np.float32),
            "unit_alignment_coverage_tgt": coverage.astype(np.float32),
            "unit_alignment_coverage_binary_tgt": coverage_binary.astype(np.float32),
            "unit_alignment_coverage_fraction_tgt": coverage_fraction.astype(np.float32),
            "unit_alignment_expected_frame_support_tgt": expected_frame_support.astype(np.float32),
            "unit_alignment_match_tgt": match_rate.astype(np.float32),
            "unit_alignment_cost_tgt": mean_cost.astype(np.float32),
            "unit_alignment_confidence_cost_term_tgt": confidence_cost_term.astype(np.float32),
            "unit_alignment_confidence_margin_term_tgt": confidence_margin_term.astype(np.float32),
            "unit_alignment_confidence_type_term_tgt": confidence_type_term.astype(np.float32),
            "unit_alignment_confidence_match_term_tgt": confidence_match_term.astype(np.float32),
            "unit_alignment_unmatched_speech_ratio_tgt": np.asarray(
                [float(projection.get("unmatched_speech_ratio", 0.0))],
                dtype=np.float32,
            ),
            "unit_alignment_mean_local_confidence_speech_tgt": np.asarray(
                [float(projection.get("mean_local_confidence_speech", 0.0))],
                dtype=np.float32,
            ),
            "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray(
                [float(projection.get("mean_coarse_confidence_speech", 0.0))],
                dtype=np.float32,
            ),
            "unit_alignment_mode_id_tgt": np.asarray(
                [alignment_mode_id],
                dtype=np.int64,
            ),
            "unit_alignment_kind_tgt": np.asarray(
                [alignment_kind],
                dtype=object,
            ),
            "unit_alignment_is_continuous_tgt": np.asarray(
                [1.0 if alignment_kind.startswith("continuous") else 0.0],
                dtype=np.float32,
            ),
            "alignment_source": str(
                projection.get("alignment_source", (precomputed_alignment or {}).get("alignment_source", "")) or ""
            ),
            "alignment_version": str(
                projection.get("alignment_version", (precomputed_alignment or {}).get("alignment_version", "")) or ""
            ),
            "unit_alignment_source_tgt": np.asarray(
                [
                    str(
                        projection.get("alignment_source", (precomputed_alignment or {}).get("alignment_source", ""))
                        or ""
                    )
                ],
                dtype=object,
            ),
            "unit_alignment_version_tgt": np.asarray(
                [
                    str(
                        projection.get(
                            "alignment_version",
                            (precomputed_alignment or {}).get("alignment_version", ""),
                        )
                        or ""
                    )
                ],
                dtype=object,
            ),
            "unit_alignment_source_cache_signature_tgt": np.asarray(
                [
                    (
                        self._normalize_signature_text(
                            paired_target_conditioning.get("source_cache_meta_signature")
                        )
                        if self._normalize_signature_text(
                            paired_target_conditioning.get("source_cache_meta_signature")
                        )
                        != "missing"
                        else duration_v3_cache_meta_signature(source_cache)
                    )
                    if isinstance(paired_target_conditioning, dict)
                    else duration_v3_cache_meta_signature(source_cache)
                ],
                dtype=object,
            ),
            "unit_alignment_target_cache_signature_tgt": np.asarray(
                [
                    self._normalize_signature_text(
                        paired_target_conditioning.get("paired_target_cache_meta_signature")
                    )
                    if isinstance(paired_target_conditioning, dict)
                    else "missing"
                ],
                dtype=object,
            ),
            "unit_alignment_sidecar_signature_tgt": np.asarray(
                [
                    str(
                        _extract_object_scalar(
                            paired_target_conditioning.get("paired_target_alignment_sidecar_signature")
                        )
                        or ""
                    )
                    if isinstance(paired_target_conditioning, dict)
                    else ""
                ],
                dtype=object,
            ),
        }
        out.update(
            self._build_alignment_provenance_exports(
                alignment_kind=alignment_kind,
                projection=projection,
                paired_target_conditioning=paired_target_conditioning,
            )
        )
        if self._is_enabled_flag(self.hparams.get("rhythm_v3_minimal_v1_profile", False)):
            self._validate_live_minimal_continuous_target_exports(
                out=out,
                context=item_name,
            )
        return out

    def _merge_duration_v3_rhythm_targets(self, item, source_cache, paired_target_conditioning, sample):
        minimal_v1_profile = self._is_enabled_flag(
            self.hparams.get("rhythm_v3_minimal_v1_profile", False)
        )
        use_continuous_alignment = bool(
            self.hparams.get("rhythm_v3_use_continuous_alignment", False)
        )
        if isinstance(sample, dict) and "unit_duration_tgt" in sample:
            if minimal_v1_profile and use_continuous_alignment:
                self._validate_cached_minimal_continuous_target(
                    sample,
                    context="cached unit_duration_tgt",
                )
                self._validate_cached_minimal_continuous_signatures(
                    sample=sample,
                    source_cache=source_cache if isinstance(source_cache, dict) else None,
                    paired_target_conditioning=(
                        paired_target_conditioning if isinstance(paired_target_conditioning, dict) else None
                    ),
                    context="cached unit_duration_tgt",
                )
            out = {
                "unit_duration_tgt": np.asarray(sample["unit_duration_tgt"], dtype=np.float32),
            }
            if "unit_duration_proj_raw_tgt" in sample:
                out["unit_duration_proj_raw_tgt"] = np.asarray(sample["unit_duration_proj_raw_tgt"], dtype=np.float32)
            if "unit_logstretch_proj_raw_tgt" in sample:
                out["unit_logstretch_proj_raw_tgt"] = np.asarray(sample["unit_logstretch_proj_raw_tgt"], dtype=np.float32)
            if "unit_confidence_local_tgt" in sample:
                out["unit_confidence_local_tgt"] = np.asarray(sample["unit_confidence_local_tgt"], dtype=np.float32)
            if "unit_confidence_coarse_tgt" in sample:
                out["unit_confidence_coarse_tgt"] = np.asarray(sample["unit_confidence_coarse_tgt"], dtype=np.float32)
            if "unit_confidence_tgt" in sample:
                out["unit_confidence_tgt"] = np.asarray(sample["unit_confidence_tgt"], dtype=np.float32)
            if "unit_confidence_tgt" in out:
                out.setdefault("unit_confidence_local_tgt", out["unit_confidence_tgt"])
                out.setdefault("unit_confidence_coarse_tgt", out["unit_confidence_tgt"])
            elif "unit_confidence_coarse_tgt" in out:
                out["unit_confidence_tgt"] = np.asarray(out["unit_confidence_coarse_tgt"], dtype=np.float32)
                out.setdefault("unit_confidence_local_tgt", out["unit_confidence_coarse_tgt"])
            for key in (
                "unit_alignment_coverage_tgt",
                "unit_alignment_coverage_binary_tgt",
                "unit_alignment_coverage_fraction_tgt",
                "unit_alignment_expected_frame_support_tgt",
                "unit_alignment_match_tgt",
                "unit_alignment_cost_tgt",
                "unit_alignment_confidence_cost_term_tgt",
                "unit_alignment_confidence_margin_term_tgt",
                "unit_alignment_confidence_type_term_tgt",
                "unit_alignment_confidence_match_term_tgt",
                "unit_alignment_unmatched_speech_ratio_tgt",
                "unit_alignment_mean_local_confidence_speech_tgt",
                "unit_alignment_mean_coarse_confidence_speech_tgt",
                "unit_alignment_mode_id_tgt",
                "unit_alignment_is_continuous_tgt",
            ):
                if key in sample:
                    dtype = np.int64 if key.endswith("_mode_id_tgt") else np.float32
                    out[key] = np.asarray(sample[key], dtype=dtype)
            alignment_mode_id = out.get("unit_alignment_mode_id_tgt", sample.get("unit_alignment_mode_id_tgt"))
            out["unit_alignment_kind_tgt"] = np.asarray(
                [
                    self._normalize_alignment_kind_export(
                        sample.get("unit_alignment_kind_tgt"),
                        mode_id=alignment_mode_id,
                    )
                ],
                dtype=object,
            )
            alignment_source_value = (
                sample.get("unit_alignment_source_tgt")
                if "unit_alignment_source_tgt" in sample
                else sample.get("alignment_source")
            )
            alignment_version_value = (
                sample.get("unit_alignment_version_tgt")
                if "unit_alignment_version_tgt" in sample
                else sample.get("alignment_version")
            )
            out["alignment_source"] = str(_extract_object_scalar(alignment_source_value) or "")
            out["alignment_version"] = str(_extract_object_scalar(alignment_version_value) or "")
            out["unit_alignment_source_tgt"] = np.asarray([out["alignment_source"]], dtype=object)
            out["unit_alignment_version_tgt"] = np.asarray([out["alignment_version"]], dtype=object)
            alignment_kind = str(np.asarray(out["unit_alignment_kind_tgt"], dtype=object).reshape(-1)[0])
            out["unit_alignment_is_continuous_tgt"] = np.asarray(
                [1.0 if alignment_kind.startswith("continuous") else 0.0],
                dtype=np.float32,
            )
            for key in self._alignment_provenance_optional_keys():
                if key in sample:
                    dtype = object if key in _ALIGNMENT_PROVENANCE_OBJECT_KEYS else np.float32
                    out[key] = np.asarray(sample[key], dtype=dtype)
            synthesized_provenance = self._build_alignment_provenance_exports(
                alignment_kind=alignment_kind,
                paired_target_conditioning=sample,
            )
            if minimal_v1_profile and use_continuous_alignment:
                missing_required = [
                    key
                    for key in _MINIMAL_CONTINUOUS_SIGNATURE_KEYS
                    if key not in out
                ]
                if missing_required:
                    raise RuntimeError(
                        "minimal_v1_profile cached paired-target supervision must preserve cached continuous provenance; "
                        "missing fields: " + ", ".join(missing_required)
                    )
                for key, value in synthesized_provenance.items():
                    if key not in out and key not in _MINIMAL_CONTINUOUS_SIGNATURE_KEYS:
                        out[key] = value
            else:
                out.update(
                    {
                        key: out.get(key, value)
                        for key, value in synthesized_provenance.items()
                    }
                )
            out.setdefault("alignment_source", "")
            out.setdefault("alignment_version", "")
            return out

        allow_source_self_target_fallback = self._is_enabled_flag(
            self.hparams.get("rhythm_v3_allow_source_self_target_fallback", False)
        )
        if minimal_v1_profile and allow_source_self_target_fallback:
            raise RuntimeError(
                "minimal_v1_profile forbids source-self target fallback; "
                "provide paired target projection or explicit cached unit_duration_tgt."
            )
        require_same_text = bool(self._require_same_text_paired_target())
        source_item_name = str(item.get("item_name", "<unknown-item>")) if isinstance(item, dict) else "<unknown-item>"
        if isinstance(paired_target_conditioning, dict):
            paired_target_item_name = paired_target_conditioning.get("paired_target_item_name")
            if paired_target_item_name is not None:
                paired_target_item_name = str(_extract_object_scalar(paired_target_item_name))
                if (
                    paired_target_item_name == source_item_name
                    and not allow_source_self_target_fallback
                ):
                    raise RuntimeError(
                        "Canonical duration_v3 training requires an external paired target item or explicit unit_duration_tgt. "
                        "Self paired-target projection is disabled unless rhythm_v3_allow_source_self_target_fallback=true."
                    )
            paired_target_text_signature = paired_target_conditioning.get("paired_target_text_signature")
            source_text_signature = paired_target_conditioning.get("source_text_signature")
            if require_same_text:
                paired_sig = _extract_object_scalar(paired_target_text_signature)
                source_sig = _extract_object_scalar(source_text_signature)
                if paired_sig is None or source_sig is None or paired_sig != source_sig:
                    raise RuntimeError(
                        (
                            "minimal_v1 requires same-text paired target projection or explicit cached unit_duration_tgt."
                            if minimal_v1_profile
                            else "V1 paired-target projection requires same-text paired target."
                        )
                    )
            if self._disallow_same_text_paired_target():
                if paired_target_text_signature is not None and source_text_signature is not None:
                    paired_sig = _extract_object_scalar(paired_target_text_signature)
                    source_sig = _extract_object_scalar(source_text_signature)
                    if paired_sig is not None and source_sig is not None and paired_sig == source_sig:
                        raise RuntimeError(
                            "rhythm_v3 paired-target projection forbids same-text targets by default. "
                            "Provide a different-text paired target or set rhythm_v3_disallow_same_text_paired_target=false."
                        )

        paired = self._build_paired_duration_v3_targets(
            item=item,
            source_cache=source_cache,
            paired_target_conditioning=paired_target_conditioning,
        )
        if paired is not None:
            if minimal_v1_profile:
                self._enforce_minimal_continuous_alignment_gate(
                    alignment_kind=paired.get("unit_alignment_kind_tgt", paired.get("alignment_kind", "")),
                    unmatched_speech_ratio=_extract_object_scalar(
                        paired.get("unit_alignment_unmatched_speech_ratio_tgt")
                    )
                    or 0.0,
                    mean_local_confidence_speech=_extract_object_scalar(
                        paired.get("unit_alignment_mean_local_confidence_speech_tgt")
                    )
                    or 0.0,
                    mean_coarse_confidence_speech=_extract_object_scalar(
                        paired.get("unit_alignment_mean_coarse_confidence_speech_tgt")
                    )
                    or 0.0,
                    context=source_item_name,
                )
            return paired

        if allow_source_self_target_fallback:
            target = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
            return {
                "unit_duration_tgt": target,
                "unit_confidence_local_tgt": np.ones_like(target, dtype=np.float32),
                "unit_confidence_coarse_tgt": np.ones_like(target, dtype=np.float32),
                "unit_confidence_tgt": np.ones_like(target, dtype=np.float32),
                "unit_alignment_coverage_tgt": np.ones_like(target, dtype=np.float32),
                "unit_alignment_match_tgt": np.ones_like(target, dtype=np.float32),
                "unit_alignment_cost_tgt": np.zeros_like(target, dtype=np.float32),
                "unit_alignment_unmatched_speech_ratio_tgt": np.asarray([0.0], dtype=np.float32),
                "unit_alignment_mean_local_confidence_speech_tgt": np.asarray([1.0], dtype=np.float32),
                "unit_alignment_mean_coarse_confidence_speech_tgt": np.asarray([1.0], dtype=np.float32),
                "unit_alignment_kind_tgt": np.asarray(["discrete"], dtype=object),
                "unit_alignment_is_continuous_tgt": np.asarray([0.0], dtype=np.float32),
                "unit_alignment_source_tgt": np.asarray([""], dtype=object),
                "unit_alignment_version_tgt": np.asarray([""], dtype=object),
                "alignment_source": "",
                "alignment_version": "",
                **self._build_alignment_provenance_exports(
                    alignment_kind=_ALIGNMENT_KIND_DISCRETE,
                ),
            }

        raise RuntimeError(
            "Canonical duration_v3 training now requires paired run projection/alignment targets. "
            "Provide unit_duration_tgt explicitly or supply an external paired-target run lattice that can be projected onto the source lattice."
        )

__all__ = ["DurationV3DatasetMixin", "_align_target_runs_to_source_discrete"]
