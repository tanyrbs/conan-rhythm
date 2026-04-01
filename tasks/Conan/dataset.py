
from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
import numpy as np
from modules.Conan.rhythm.supervision import (
    RHYTHM_CACHE_VERSION,
    RHYTHM_GUIDANCE_SURFACE_NAME,
    RHYTHM_REFERENCE_MODE_STATIC_REF_FULL,
    RHYTHM_RETIMED_SOURCE_GUIDANCE,
    RHYTHM_RETIMED_SOURCE_TEACHER,
    RHYTHM_TEACHER_SURFACE_NAME,
    RHYTHM_TRACE_HOP_MS,
    RHYTHM_UNIT_HOP_MS,
    build_reference_guided_targets,
    build_reference_rhythm_conditioning,
    build_reference_teacher_targets,
    build_retimed_mel_target,
    build_source_rhythm_cache,
)

class ConanDataset(FastSpeechDataset):
    _RHYTHM_SOURCE_CACHE_KEYS = (
        "content_units",
        "dur_anchor_src",
        "open_run_mask",
        "sep_hint",
    )
    _RHYTHM_REF_CACHE_KEYS = (
        "ref_rhythm_stats",
        "ref_rhythm_trace",
    )
    _RHYTHM_TARGET_KEYS = (
        "rhythm_speech_exec_tgt",
        "rhythm_pause_exec_tgt",
        "rhythm_speech_budget_tgt",
        "rhythm_pause_budget_tgt",
        "rhythm_guidance_speech_tgt",
        "rhythm_guidance_pause_tgt",
        "rhythm_teacher_speech_exec_tgt",
        "rhythm_teacher_pause_exec_tgt",
        "rhythm_teacher_speech_budget_tgt",
        "rhythm_teacher_pause_budget_tgt",
    )
    _RHYTHM_META_KEYS = (
        "rhythm_cache_version",
        "rhythm_unit_hop_ms",
        "rhythm_trace_hop_ms",
        "rhythm_trace_bins",
        "rhythm_trace_horizon",
        "rhythm_reference_mode_id",
        "rhythm_target_confidence",
        "rhythm_guidance_confidence",
        "rhythm_teacher_confidence",
        "rhythm_retimed_target_source_id",
        "rhythm_retimed_target_confidence",
    )

    def _resolve_rhythm_target_mode(self) -> str:
        mode = str(self.hparams.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
        aliases = {
            "auto": "prefer_cache",
            "offline": "cached_only",
            "offline_only": "cached_only",
            "never": "cached_only",
            "runtime": "runtime_only",
            "always": "runtime_only",
        }
        return aliases.get(mode, mode)

    def _expected_rhythm_cache_version(self) -> int:
        return int(self.hparams.get("rhythm_cache_version", RHYTHM_CACHE_VERSION))

    @staticmethod
    def _extract_scalar(value):
        arr = np.asarray(value)
        if arr.size <= 0:
            raise RuntimeError("Rhythm cache metadata contains an empty scalar field.")
        return arr.reshape(-1)[0]

    @staticmethod
    def _missing_keys(item, keys):
        return [key for key in keys if key not in item]

    def _require_cached_keys(self, *, item, keys, item_name: str, reason: str):
        missing = self._missing_keys(item, keys)
        if missing:
            raise RuntimeError(
                f"Rhythm cached_only requires {reason} in {item_name}, missing keys: {missing}"
            )

    def _validate_rhythm_cache_version(self, item, *, item_name: str):
        if "rhythm_cache_version" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_cache_version in {item_name}. Re-binarize the dataset."
            )
        found = int(self._extract_scalar(item["rhythm_cache_version"]))
        expected = self._expected_rhythm_cache_version()
        if found != expected:
            raise RuntimeError(
                f"Rhythm cache version mismatch in {item_name}: found={found}, expected={expected}. Re-binarize the dataset."
            )

    def _validate_rhythm_cache_contract(self, item, *, item_name: str):
        self._validate_rhythm_cache_version(item, item_name=item_name)
        expected_numeric = {
            "rhythm_unit_hop_ms": int(self.hparams.get("rhythm_unit_hop_ms", RHYTHM_UNIT_HOP_MS)),
            "rhythm_trace_hop_ms": int(self.hparams.get("rhythm_trace_hop_ms", RHYTHM_TRACE_HOP_MS)),
            "rhythm_trace_bins": int(self.hparams.get("rhythm_trace_bins", 24)),
            "rhythm_trace_horizon": float(self.hparams.get("rhythm_trace_horizon", 0.35)),
            "rhythm_reference_mode_id": int(
                self.hparams.get("rhythm_reference_mode_id", RHYTHM_REFERENCE_MODE_STATIC_REF_FULL)
            ),
        }
        for key, expected in expected_numeric.items():
            if key not in item:
                raise RuntimeError(
                    f"Rhythm cached_only requires {key} in {item_name}. Re-binarize the dataset."
                )
            found = self._extract_scalar(item[key])
            if isinstance(expected, float):
                if abs(float(found) - expected) > 1e-5:
                    raise RuntimeError(
                        f"Rhythm cache contract mismatch in {item_name} for {key}: "
                        f"found={float(found):.6f}, expected={expected:.6f}. Re-binarize the dataset."
                    )
            else:
                if int(found) != expected:
                    raise RuntimeError(
                        f"Rhythm cache contract mismatch in {item_name} for {key}: "
                        f"found={int(found)}, expected={expected}. Re-binarize the dataset."
                    )
        if "rhythm_guidance_surface_name" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_guidance_surface_name in {item_name}. Re-binarize the dataset."
            )
        guidance_name = str(self._extract_scalar(item["rhythm_guidance_surface_name"]))
        if guidance_name != RHYTHM_GUIDANCE_SURFACE_NAME:
            raise RuntimeError(
                f"Rhythm guidance surface mismatch in {item_name}: "
                f"found={guidance_name}, expected={RHYTHM_GUIDANCE_SURFACE_NAME}. Re-binarize the dataset."
            )

    def _required_cached_target_keys(self):
        keys = [
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_speech_budget_tgt",
            "rhythm_pause_budget_tgt",
            "rhythm_cache_version",
            "rhythm_unit_hop_ms",
            "rhythm_trace_hop_ms",
            "rhythm_trace_bins",
            "rhythm_trace_horizon",
            "rhythm_reference_mode_id",
            "rhythm_target_confidence",
            "rhythm_guidance_confidence",
            "rhythm_guidance_surface_name",
        ]
        if float(self.hparams.get("lambda_rhythm_guidance", 0.0)) > 0:
            keys.extend([
                "rhythm_guidance_speech_tgt",
                "rhythm_guidance_pause_tgt",
            ])
        need_teacher = (
            float(self.hparams.get("lambda_rhythm_distill", 0.0)) > 0
            or bool(self.hparams.get("rhythm_require_cached_teacher", False))
            or (
                bool(self.hparams.get("rhythm_use_retimed_target_if_available", False))
                and str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance")).strip().lower() == "teacher"
            )
        )
        if need_teacher:
            keys.extend([
                "rhythm_teacher_speech_exec_tgt",
                "rhythm_teacher_pause_exec_tgt",
                "rhythm_teacher_speech_budget_tgt",
                "rhythm_teacher_pause_budget_tgt",
                "rhythm_teacher_confidence",
                "rhythm_teacher_surface_name",
            ])
        if bool(self.hparams.get("rhythm_require_retimed_cache", False)):
            keys.extend([
                "rhythm_retimed_mel_tgt",
                "rhythm_retimed_mel_len",
                "rhythm_retimed_frame_weight",
                "rhythm_retimed_target_source_id",
                "rhythm_retimed_target_confidence",
                "rhythm_retimed_target_surface_name",
            ])
        return tuple(dict.fromkeys(keys))

    def _validate_retimed_cache_contract(self, item, *, item_name: str):
        if "rhythm_retimed_target_source_id" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_retimed_target_source_id in {item_name}. Re-binarize the dataset."
            )
        found_source_id = int(self._extract_scalar(item["rhythm_retimed_target_source_id"]))
        expected_source = str(self.hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
        expected_source_id = RHYTHM_RETIMED_SOURCE_TEACHER if expected_source == "teacher" else RHYTHM_RETIMED_SOURCE_GUIDANCE
        if found_source_id != expected_source_id:
            raise RuntimeError(
                f"Rhythm retimed cache source mismatch in {item_name}: "
                f"found={found_source_id}, expected={expected_source_id}. Re-binarize the dataset."
            )
        if "rhythm_retimed_target_surface_name" not in item:
            raise RuntimeError(
                f"Rhythm cached_only requires rhythm_retimed_target_surface_name in {item_name}. Re-binarize the dataset."
            )
        found_surface = str(self._extract_scalar(item["rhythm_retimed_target_surface_name"]))
        expected_surface = RHYTHM_TEACHER_SURFACE_NAME if expected_source_id == RHYTHM_RETIMED_SOURCE_TEACHER else RHYTHM_GUIDANCE_SURFACE_NAME
        if found_surface != expected_surface:
            raise RuntimeError(
                f"Rhythm retimed surface mismatch in {item_name}: "
                f"found={found_surface}, expected={expected_surface}. Re-binarize the dataset."
            )

    def _validate_source_cache_shapes(self, cache, *, item_name: str):
        lengths = {key: int(np.asarray(cache[key]).reshape(-1).shape[0]) for key in self._RHYTHM_SOURCE_CACHE_KEYS}
        for key in ("sealed_mask", "boundary_confidence"):
            if key in cache:
                lengths[key] = int(np.asarray(cache[key]).reshape(-1).shape[0])
        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                f"Rhythm source cache shape mismatch in {item_name}: {lengths}. Re-binarize the dataset."
            )

    def _validate_reference_conditioning_shapes(self, conditioning, *, item_name: str):
        stats = np.asarray(conditioning["ref_rhythm_stats"])
        trace = np.asarray(conditioning["ref_rhythm_trace"])
        stats_dim = int(self.hparams.get("rhythm_stats_dim", 6))
        trace_bins = int(self.hparams.get("rhythm_trace_bins", 24))
        trace_dim = int(self.hparams.get("rhythm_trace_dim", 5))
        if stats.reshape(-1).shape[0] != stats_dim:
            raise RuntimeError(
                f"Rhythm stats shape mismatch in {item_name}: found={tuple(stats.shape)}, expected_last_dim={stats_dim}."
            )
        if trace.ndim != 2 or trace.shape[0] != trace_bins or trace.shape[1] != trace_dim:
            raise RuntimeError(
                f"Rhythm trace shape mismatch in {item_name}: found={tuple(trace.shape)}, expected=({trace_bins}, {trace_dim})."
            )

    def _validate_target_shapes(self, targets, *, item_name: str, expected_units: int):
        unit_keys = [
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_guidance_speech_tgt",
            "rhythm_guidance_pause_tgt",
            "rhythm_teacher_speech_exec_tgt",
            "rhythm_teacher_pause_exec_tgt",
        ]
        for key in unit_keys:
            if key not in targets:
                continue
            found = int(np.asarray(targets[key]).reshape(-1).shape[0])
            if found != expected_units:
                raise RuntimeError(
                    f"Rhythm target length mismatch in {item_name} for {key}: "
                    f"found={found}, expected={expected_units}."
                )
        budget_keys = [
            "rhythm_speech_budget_tgt",
            "rhythm_pause_budget_tgt",
            "rhythm_teacher_speech_budget_tgt",
            "rhythm_teacher_pause_budget_tgt",
        ]
        for key in budget_keys:
            if key not in targets:
                continue
            found = int(np.asarray(targets[key]).reshape(-1).shape[0])
            if found != 1:
                raise RuntimeError(
                    f"Rhythm budget target shape mismatch in {item_name} for {key}: expected scalar/[1], found={tuple(np.asarray(targets[key]).shape)}."
                )
        if "rhythm_retimed_mel_tgt" in targets or "rhythm_retimed_mel_len" in targets or "rhythm_retimed_frame_weight" in targets:
            required = ("rhythm_retimed_mel_tgt", "rhythm_retimed_mel_len", "rhythm_retimed_frame_weight")
            missing = [key for key in required if key not in targets]
            if missing:
                raise RuntimeError(
                    f"Rhythm retimed cache incomplete in {item_name}, missing={missing}. Re-binarize the dataset."
                )
            mel = np.asarray(targets["rhythm_retimed_mel_tgt"])
            frame_weight = np.asarray(targets["rhythm_retimed_frame_weight"]).reshape(-1)
            mel_len = int(self._extract_scalar(targets["rhythm_retimed_mel_len"]))
            if mel.ndim != 2:
                raise RuntimeError(
                    f"Rhythm retimed mel target must be rank-2 in {item_name}, found shape={tuple(mel.shape)}."
                )
            if mel.shape[0] != mel_len or frame_weight.shape[0] != mel_len:
                raise RuntimeError(
                    f"Rhythm retimed cache length mismatch in {item_name}: mel_len={mel_len}, "
                    f"mel_frames={mel.shape[0]}, weight_frames={frame_weight.shape[0]}."
                )

    def _should_use_self_rhythm_reference(self, item, *, target_mode: str) -> bool:
        policy = str(self.hparams.get("rhythm_cached_reference_policy", "self") or "self").strip().lower()
        if policy in {"sample_ref", "paired", "external"}:
            return False
        has_cached = all(key in item for key in self._RHYTHM_REF_CACHE_KEYS) and any(
            key in item for key in ("rhythm_speech_exec_tgt", "rhythm_pause_exec_tgt")
        )
        return target_mode != "runtime_only" and has_cached

    def _adapt_cached_targets_to_prefix(self, *, item, cached_targets, source_cache, sample):
        visible_units = int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])
        full_units = int(np.asarray(item["dur_anchor_src"]).reshape(-1).shape[0]) if "dur_anchor_src" in item else visible_units
        if visible_units >= full_units:
            return dict(cached_targets)
        adapted = dict(cached_targets)
        unit_keys = [
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_guidance_speech_tgt",
            "rhythm_guidance_pause_tgt",
            "rhythm_teacher_speech_exec_tgt",
            "rhythm_teacher_pause_exec_tgt",
        ]
        for key in unit_keys:
            if key in adapted:
                adapted[key] = np.asarray(adapted[key]).reshape(-1)[:visible_units].astype(np.float32)
        if "rhythm_speech_exec_tgt" in adapted:
            adapted["rhythm_speech_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_speech_exec_tgt"]).sum())], dtype=np.float32
            )
        if "rhythm_pause_exec_tgt" in adapted:
            adapted["rhythm_pause_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_pause_exec_tgt"]).sum())], dtype=np.float32
            )
        if "rhythm_teacher_speech_exec_tgt" in adapted:
            adapted["rhythm_teacher_speech_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_teacher_speech_exec_tgt"]).sum())], dtype=np.float32
            )
        if "rhythm_teacher_pause_exec_tgt" in adapted:
            adapted["rhythm_teacher_pause_budget_tgt"] = np.asarray(
                [float(np.asarray(adapted["rhythm_teacher_pause_exec_tgt"]).sum())], dtype=np.float32
            )
        if "rhythm_retimed_mel_tgt" in adapted:
            source_id = int(np.asarray(adapted.get("rhythm_retimed_target_source_id", [RHYTHM_RETIMED_SOURCE_GUIDANCE])).reshape(-1)[0])
            if source_id == RHYTHM_RETIMED_SOURCE_TEACHER and "rhythm_teacher_speech_exec_tgt" in adapted:
                speech_key = "rhythm_teacher_speech_exec_tgt"
                pause_key = "rhythm_teacher_pause_exec_tgt"
            else:
                speech_key = "rhythm_speech_exec_tgt"
                pause_key = "rhythm_pause_exec_tgt"
            adapted.update(
                build_retimed_mel_target(
                    mel=sample["mel"].cpu().numpy(),
                    dur_anchor_src=source_cache["dur_anchor_src"],
                    speech_exec_tgt=adapted[speech_key],
                    pause_exec_tgt=adapted[pause_key],
                    unit_mask=(np.asarray(source_cache["dur_anchor_src"]) > 0).astype(np.float32),
                    pause_frame_weight=float(self.hparams.get("rhythm_retimed_pause_frame_weight", 0.20)),
                    stretch_weight_min=float(self.hparams.get("rhythm_retimed_stretch_weight_min", 0.35)),
                )
            )
        return adapted

    def _get_source_rhythm_cache(self, item, visible_tokens, *, target_mode: str):
        cache_keys = self._RHYTHM_SOURCE_CACHE_KEYS
        full_tokens = np.asarray(item["hubert"])
        visible_tokens = np.asarray(visible_tokens)
        if all(key in item for key in cache_keys):
            cache = {key: item[key] for key in cache_keys}
            self._validate_source_cache_shapes(
                cache,
                item_name=str(item.get("item_name", "<unknown-item>")),
            )
            if target_mode == "cached_only":
                item_name = str(item.get("item_name", "<unknown-item>"))
                self._validate_rhythm_cache_contract(item, item_name=item_name)
                self._require_cached_keys(
                    item=item,
                    keys=cache_keys,
                    item_name=item_name,
                    reason="source rhythm cache",
                )
                self._validate_source_cache_shapes(cache, item_name=item_name)
            if int(full_tokens.shape[0]) == int(visible_tokens.shape[0]):
                return cache
        if target_mode == "cached_only":
            # Prefix rebuild is deterministic from visible tokens and avoids false failures when
            # dataset sampling crops the source sequence.
            return build_source_rhythm_cache(
                visible_tokens,
                silent_token=self.hparams.get("silent_token", 57),
                separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            )
        return build_source_rhythm_cache(
            visible_tokens,
            silent_token=self.hparams.get("silent_token", 57),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
        )

    def _get_reference_rhythm_conditioning(self, ref_item, sample, *, target_mode: str):
        cache_keys = self._RHYTHM_REF_CACHE_KEYS
        if ref_item is not None and all(key in ref_item for key in cache_keys):
            conditioning = {key: ref_item[key] for key in cache_keys}
            if target_mode == "cached_only":
                item_name = str(ref_item.get("item_name", "<unknown-ref-item>"))
                self._validate_rhythm_cache_contract(ref_item, item_name=item_name)
                self._require_cached_keys(
                    item=ref_item,
                    keys=cache_keys,
                    item_name=item_name,
                    reason="reference rhythm cache",
                )
            self._validate_reference_conditioning_shapes(
                conditioning,
                item_name=str(ref_item.get("item_name", "<unknown-ref-item>")),
            )
            return conditioning
        if target_mode == "cached_only":
            item_name = str(ref_item.get("item_name", "<unknown-ref-item>")) if ref_item is not None else "<missing-ref-item>"
            if ref_item is None:
                raise RuntimeError("Rhythm cached_only requires cached reference conditioning, but ref_item is missing.")
            raise RuntimeError(
                f"Rhythm cached_only requires cached reference conditioning in {item_name}. Re-binarize the dataset."
            )
        conditioning = build_reference_rhythm_conditioning(
            sample["ref_mel"],
            trace_bins=int(self.hparams.get("rhythm_trace_bins", 24)),
            trace_horizon=float(self.hparams.get("rhythm_trace_horizon", 0.35)),
        )
        self._validate_reference_conditioning_shapes(conditioning, item_name="<runtime-ref-conditioning>")
        return conditioning

    def _build_runtime_rhythm_targets(self, source_cache, ref_conditioning):
        unit_mask = (np.asarray(source_cache["dur_anchor_src"]) > 0).astype(np.float32)
        shared_kwargs = dict(
            dur_anchor_src=source_cache["dur_anchor_src"],
            unit_mask=unit_mask,
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=ref_conditioning["ref_rhythm_trace"],
            rate_scale_min=float(self.hparams.get("rhythm_guidance_rate_scale_min", 0.60)),
            rate_scale_max=float(self.hparams.get("rhythm_guidance_rate_scale_max", 1.80)),
            local_rate_strength=float(self.hparams.get("rhythm_guidance_local_rate_strength", 0.35)),
            segment_bias_strength=float(self.hparams.get("rhythm_guidance_segment_bias_strength", 0.25)),
            pause_strength=float(self.hparams.get("rhythm_guidance_pause_strength", 1.00)),
            boundary_strength=float(self.hparams.get("rhythm_guidance_boundary_strength", 1.25)),
            pause_budget_ratio_cap=float(self.hparams.get("rhythm_guidance_pause_budget_ratio_cap", 0.75)),
        )
        teacher_kwargs = dict(
            dur_anchor_src=source_cache["dur_anchor_src"],
            unit_mask=unit_mask,
            source_boundary_cue=source_cache.get("boundary_confidence"),
            ref_rhythm_stats=ref_conditioning["ref_rhythm_stats"],
            ref_rhythm_trace=ref_conditioning["ref_rhythm_trace"],
            rate_scale_min=float(self.hparams.get("rhythm_teacher_rate_scale_min", 0.55)),
            rate_scale_max=float(self.hparams.get("rhythm_teacher_rate_scale_max", 1.95)),
            local_rate_strength=float(self.hparams.get("rhythm_teacher_local_rate_strength", 0.45)),
            segment_bias_strength=float(self.hparams.get("rhythm_teacher_segment_bias_strength", 0.30)),
            pause_strength=float(self.hparams.get("rhythm_teacher_pause_strength", 1.10)),
            boundary_strength=float(self.hparams.get("rhythm_teacher_boundary_strength", 1.50)),
            pause_budget_ratio_cap=float(self.hparams.get("rhythm_teacher_pause_budget_ratio_cap", 0.80)),
            speech_smooth_kernel=int(self.hparams.get("rhythm_teacher_speech_smooth_kernel", 3)),
            pause_topk_ratio=float(self.hparams.get("rhythm_teacher_pause_topk_ratio", 0.30)),
        )
        targets = {}
        if self.hparams.get("rhythm_dataset_build_guidance_from_ref", True):
            targets.update(build_reference_guided_targets(**shared_kwargs))
        if self.hparams.get("rhythm_dataset_build_teacher_from_ref", False):
            targets.update(build_reference_teacher_targets(**teacher_kwargs))
        return targets

    def _merge_rhythm_targets(self, item, source_cache, ref_conditioning, sample):
        target_mode = self._resolve_rhythm_target_mode()
        cached_targets = {key: item[key] for key in self._RHYTHM_TARGET_KEYS if key in item}
        cached_targets.update({key: item[key] for key in self._RHYTHM_META_KEYS if key in item})
        if target_mode == "cached_only":
            item_name = str(item.get("item_name", "<unknown-item>"))
            self._validate_rhythm_cache_contract(item, item_name=item_name)
            self._require_cached_keys(
                item=item,
                keys=self._required_cached_target_keys(),
                item_name=item_name,
                reason="rhythm targets/meta cache",
            )
            if "rhythm_teacher_surface_name" in item:
                teacher_name = str(self._extract_scalar(item["rhythm_teacher_surface_name"]))
                if teacher_name != RHYTHM_TEACHER_SURFACE_NAME:
                    raise RuntimeError(
                        f"Rhythm teacher surface mismatch in {item_name}: "
                        f"found={teacher_name}, expected={RHYTHM_TEACHER_SURFACE_NAME}. Re-binarize the dataset."
                    )
            if bool(self.hparams.get("rhythm_require_retimed_cache", False)):
                self._validate_retimed_cache_contract(item, item_name=item_name)
            cached_targets = self._adapt_cached_targets_to_prefix(
                item=item,
                cached_targets=cached_targets,
                source_cache=source_cache,
                sample=sample,
            )
            self._validate_target_shapes(
                cached_targets,
                item_name=item_name,
                expected_units=int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0]),
            )
            return cached_targets

        runtime_targets = self._build_runtime_rhythm_targets(source_cache, ref_conditioning)
        if target_mode == "runtime_only":
            return runtime_targets

        merged = dict(cached_targets)
        for key, value in runtime_targets.items():
            merged.setdefault(key, value)
        if "rhythm_speech_exec_tgt" in merged:
            merged = self._adapt_cached_targets_to_prefix(
                item=item,
                cached_targets=merged,
                source_cache=source_cache,
                sample=sample,
            )
            self._validate_target_shapes(
                merged,
                item_name=str(item.get("item_name", "<unknown-item>")),
                expected_units=int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0]),
            )
        return merged

    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(ConanDataset, self).__getitem__(index)
        item = self._get_item(index)
        ref_item = None
        if "ref_item_id" in sample:
            ref_item = self._get_item(int(sample["ref_item_id"]))
        
        # if isinstance(item['hubert'], str):
        #     # Convert string to numeric array
        #     content = [float(x) for x in item['hubert'].split()]
        #     sample["content"] = torch.LongTensor(content)
        # else:
            # Already a numeric array case
        visible_len = int(sample["mel"].shape[0])
        if isinstance(item['hubert'], str):
            content_visible = [int(float(x)) for x in item['hubert'].split()[:visible_len]]
        else:
            content_visible = list(item['hubert'][:visible_len])
        sample["content"] = torch.LongTensor(content_visible)
        target_mode = self._resolve_rhythm_target_mode()

        optional_rhythm_keys = [
            "content_units",
            "dur_anchor_src",
            "open_run_mask",
            "sealed_mask",
            "sep_hint",
            "boundary_confidence",
            "ref_rhythm_stats",
            "ref_rhythm_trace",
            "rhythm_cache_version",
            "rhythm_unit_hop_ms",
            "rhythm_trace_hop_ms",
            "rhythm_trace_bins",
            "rhythm_trace_horizon",
            "rhythm_reference_mode_id",
            "rhythm_target_confidence",
            "rhythm_guidance_confidence",
            "rhythm_teacher_confidence",
            "rhythm_retimed_target_source_id",
            "rhythm_retimed_target_confidence",
            "rhythm_speech_exec_tgt",
            "rhythm_pause_exec_tgt",
            "rhythm_speech_budget_tgt",
            "rhythm_pause_budget_tgt",
            "rhythm_guidance_speech_tgt",
            "rhythm_guidance_pause_tgt",
            "rhythm_teacher_speech_exec_tgt",
            "rhythm_teacher_pause_exec_tgt",
            "rhythm_teacher_speech_budget_tgt",
            "rhythm_teacher_pause_budget_tgt",
            "rhythm_retimed_mel_tgt",
            "rhythm_retimed_mel_len",
            "rhythm_retimed_frame_weight",
        ]
        rhythm_runtime_fields = {}
        source_cache = self._get_source_rhythm_cache(item, sample["content"].cpu().numpy(), target_mode=target_mode)
        rhythm_ref_item = item if self._should_use_self_rhythm_reference(item, target_mode=target_mode) else ref_item
        ref_conditioning = self._get_reference_rhythm_conditioning(rhythm_ref_item, sample, target_mode=target_mode)
        rhythm_runtime_fields.update(source_cache)
        rhythm_runtime_fields.update(ref_conditioning)
        rhythm_runtime_fields.update(
            self._merge_rhythm_targets(
                item,
                source_cache,
                ref_conditioning,
                sample,
            )
        )

        for key in optional_rhythm_keys:
            if key in rhythm_runtime_fields:
                value = rhythm_runtime_fields[key]
            elif key in item:
                value = item[key]
            else:
                continue
            if key == "ref_rhythm_trace":
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif "stats" in key or "budget" in key:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {"sealed_mask", "boundary_confidence"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {"rhythm_target_confidence", "rhythm_guidance_confidence", "rhythm_teacher_confidence"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {"rhythm_retimed_target_confidence", "rhythm_trace_horizon"}:
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key in {
                "rhythm_cache_version",
                "rhythm_unit_hop_ms",
                "rhythm_trace_hop_ms",
                "rhythm_trace_bins",
                "rhythm_reference_mode_id",
                "rhythm_retimed_target_source_id",
            }:
                sample[key] = torch.tensor(value, dtype=torch.long)
            elif key == "rhythm_retimed_mel_tgt":
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif key == "rhythm_retimed_mel_len":
                sample[key] = torch.tensor(value, dtype=torch.long)
            elif key == "rhythm_retimed_frame_weight":
                sample[key] = torch.tensor(value, dtype=torch.float32)
            else:
                sample[key] = torch.tensor(value)
        sample.pop("ref_item_id", None)

        # sample['content'] = torch.LongTensor(item['hubert'])
        # note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        # note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        # note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        # sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type

        # for key in ['mix_tech','falsetto_tech','breathy_tech','bubble_tech','strong_tech','weak_tech','pharyngeal_tech','vibrato_tech','glissando_tech']:
        #     if key not in item:
        #         item[key] = [2] * len(item['ph'])

        # mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        # falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        # breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        # sample['mix'],sample['falsetto'],sample['breathy']=mix,falsetto,breathy

        # bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])
        # strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])
        # weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])
        # sample['bubble'],sample['strong'],sample['weak']=bubble,strong,weak

        # pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])
        # vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])
        # glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])
        # sample['pharyngeal'],sample['vibrato'],sample['glissando'] = pharyngeal,vibrato,glissando
        
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(ConanDataset, self).collater(samples)
        content= collate_1d_or_2d([s['content'] for s in samples], 0).long()
        batch['content'] = content

        optional_collate = {
            "content_units": ("long", 0),
            "dur_anchor_src": ("long", 0),
            "open_run_mask": ("long", 0),
            "sealed_mask": ("float", 0.0),
            "sep_hint": ("long", 0),
            "boundary_confidence": ("float", 0.0),
            "ref_rhythm_stats": ("float", 0.0),
            "ref_rhythm_trace": ("float", 0.0),
            "rhythm_cache_version": ("long", 0),
            "rhythm_unit_hop_ms": ("long", 0),
            "rhythm_trace_hop_ms": ("long", 0),
            "rhythm_trace_bins": ("long", 0),
            "rhythm_trace_horizon": ("float", 0.0),
            "rhythm_reference_mode_id": ("long", 0),
            "rhythm_target_confidence": ("float", 0.0),
            "rhythm_guidance_confidence": ("float", 0.0),
            "rhythm_teacher_confidence": ("float", 0.0),
            "rhythm_retimed_target_source_id": ("long", 0),
            "rhythm_retimed_target_confidence": ("float", 0.0),
            "rhythm_speech_exec_tgt": ("float", 0.0),
            "rhythm_pause_exec_tgt": ("float", 0.0),
            "rhythm_speech_budget_tgt": ("float", 0.0),
            "rhythm_pause_budget_tgt": ("float", 0.0),
            "rhythm_guidance_speech_tgt": ("float", 0.0),
            "rhythm_guidance_pause_tgt": ("float", 0.0),
            "rhythm_teacher_speech_exec_tgt": ("float", 0.0),
            "rhythm_teacher_pause_exec_tgt": ("float", 0.0),
            "rhythm_teacher_speech_budget_tgt": ("float", 0.0),
            "rhythm_teacher_pause_budget_tgt": ("float", 0.0),
            "rhythm_retimed_mel_tgt": ("float", 0.0),
            "rhythm_retimed_mel_len": ("long", 0),
            "rhythm_retimed_frame_weight": ("float", 0.0),
        }
        for key, (dtype_name, pad_value) in optional_collate.items():
            if all(key in s for s in samples):
                value = collate_1d_or_2d([s[key] for s in samples], pad_value)
                if dtype_name == "long":
                    value = value.long()
                else:
                    value = value.float()
                batch[key] = value
        # notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        # note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        # note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        # batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        # mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        # falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        # breathy = collate_1d_or_2d([s['breathy'] for s in samples], 0.0)
        # batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy

        # bubble = collate_1d_or_2d([s['bubble'] for s in samples], 0.0)
        # strong = collate_1d_or_2d([s['strong'] for s in samples], 0.0)
        # weak = collate_1d_or_2d([s['weak'] for s in samples], 0.0)
        # batch['bubble'],batch['strong'],batch['weak']=bubble,strong,weak

        # pharyngeal = collate_1d_or_2d([s['pharyngeal'] for s in samples], 0.0)
        # vibrato = collate_1d_or_2d([s['vibrato'] for s in samples], 0.0)
        # glissando = collate_1d_or_2d([s['glissando'] for s in samples], 0.0)
        # batch['pharyngeal'],batch['vibrato'],batch['glissando'] = pharyngeal,vibrato,glissando    

        return batch
