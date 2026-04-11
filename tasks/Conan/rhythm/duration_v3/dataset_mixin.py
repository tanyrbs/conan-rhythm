from __future__ import annotations

import numpy as np
import torch

from modules.Conan.rhythm.policy import is_duration_operator_mode
from modules.Conan.rhythm.supervision import build_source_rhythm_cache
from tasks.Conan.rhythm.duration_v3.targets import build_pseudo_source_duration_context
from tasks.Conan.rhythm.duration_v3.task_config import is_duration_v3_prompt_summary_backbone


class DurationV3DatasetMixin:
    def _use_duration_v3_dataset_contract(self) -> bool:
        return bool(
            self.hparams.get("rhythm_enable_v3", False)
            or is_duration_operator_mode(self.hparams.get("rhythm_mode", ""))
        )

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
        generator = self._make_item_generator(item_name=item_name, salt="pseudo_source_duration")
        perturbed = build_pseudo_source_duration_context(
            torch.from_numpy(source_duration),
            torch.from_numpy(unit_mask),
            torch.from_numpy(sep_hint),
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
        prompt_mask = np.asarray(conditioning.get("prompt_unit_mask"), dtype=np.float32).copy()
        if prompt_mask.size <= 0:
            return conditioning
        valid = torch.from_numpy(prompt_mask).reshape(1, -1) > 0.5
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
        if not bool(keep.any().item()):
            first_valid = int(torch.nonzero(valid[0], as_tuple=False)[0].item())
            keep[:, first_valid] = True
        keep_np = keep.float().reshape(-1).cpu().numpy().astype(np.float32)
        augmented = self._copy_numpy_fields(conditioning)
        augmented["prompt_unit_mask"] = keep_np
        for key in ("prompt_duration_obs", "prompt_source_boundary_cue", "prompt_phrase_group_pos", "prompt_phrase_final_mask"):
            if key in augmented:
                augmented[key] = np.asarray(augmented[key], dtype=np.float32) * keep_np
        return augmented

    def _build_reference_prompt_unit_conditioning(self, prompt_item, *, target_mode: str):
        if prompt_item is None:
            return {}
        source_cache = None
        item_name = str(prompt_item.get("item_name", "<prompt-item>")) if isinstance(prompt_item, dict) else "<prompt-item>"
        if all(key in prompt_item for key in self._RHYTHM_REF_PROMPT_SOURCE_KEYS):
            source_cache = {key: prompt_item[key] for key in self._RHYTHM_REF_PROMPT_SOURCE_KEYS}
            if "sep_hint" in prompt_item:
                source_cache["sep_hint"] = prompt_item["sep_hint"]
            for extra_key in ("source_boundary_cue", "phrase_group_pos", "phrase_final_mask"):
                if extra_key in prompt_item:
                    source_cache[extra_key] = prompt_item[extra_key]
        elif target_mode != "cached_only" and "hubert" in prompt_item:
            source_cache = build_source_rhythm_cache(
                np.asarray(prompt_item["hubert"]),
                silent_token=self.hparams.get("silent_token", 57),
                separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
                tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
                phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
            )
        if source_cache is None:
            return {}
        prompt_content_units = np.asarray(source_cache["content_units"], dtype=np.int64)
        prompt_duration_obs = np.asarray(source_cache["dur_anchor_src"], dtype=np.float32)
        sep_hint = np.asarray(source_cache.get("sep_hint", np.zeros_like(prompt_content_units)), dtype=np.float32)
        prompt_unit_mask = (prompt_duration_obs > 0).astype(np.float32)
        if sep_hint.shape == prompt_unit_mask.shape:
            prompt_unit_mask = prompt_unit_mask * (1.0 - sep_hint.clip(0.0, 1.0))
        return self._maybe_augment_prompt_unit_conditioning({
            "prompt_content_units": prompt_content_units,
            "prompt_duration_obs": prompt_duration_obs,
            "prompt_unit_mask": prompt_unit_mask,
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
        }, item_name=item_name)

    def _merge_duration_v3_rhythm_targets(self, item, source_cache, ref_conditioning, sample):
        del item, ref_conditioning, sample
        return {
            "unit_duration_tgt": np.asarray(source_cache["dur_anchor_src"], dtype=np.float32),
        }


__all__ = ["DurationV3DatasetMixin"]
