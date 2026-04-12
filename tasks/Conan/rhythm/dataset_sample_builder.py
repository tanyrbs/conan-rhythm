from __future__ import annotations

import numpy as np
import torch


class RhythmDatasetSampleAssembler:
    def __init__(self, owner):
        self.owner = owner

    @property
    def hparams(self):
        return self.owner.hparams

    def _build_reference_conditioning(self, *, rhythm_ref_item, sample, item, target_mode: str):
        try:
            return self.owner._get_reference_rhythm_conditioning(
                rhythm_ref_item,
                sample,
                target_mode=target_mode,
                item=item,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'item'" not in str(exc):
                raise
            return self.owner._get_reference_rhythm_conditioning(
                rhythm_ref_item,
                sample,
                target_mode=target_mode,
            )

    def _tensorize_optional_value(self, key, value):
        if key in {
            "ref_rhythm_trace",
            "ref_phrase_trace",
            "planner_ref_phrase_trace",
            "slow_rhythm_memory",
            "planner_slow_rhythm_memory",
        }:
            return torch.tensor(value, dtype=torch.float32)
        if key in {
            "source_boundary_cue",
            "phrase_group_pos",
            "phrase_final_mask",
            "prompt_source_boundary_cue",
            "prompt_phrase_group_pos",
            "prompt_phrase_final_mask",
            "source_silence_mask",
            "source_run_stability",
            "rhythm_offline_source_silence_mask",
            "rhythm_offline_source_run_stability",
            "slow_rhythm_summary",
            "planner_slow_rhythm_summary",
            "selector_meta_scores",
            "rhythm_teacher_allocation_tgt",
            "rhythm_teacher_prefix_clock_tgt",
            "rhythm_teacher_prefix_backlog_tgt",
            "rhythm_offline_source_boundary_cue",
            "rhythm_offline_phrase_group_pos",
            "rhythm_offline_phrase_final_mask",
            "rhythm_stream_prefix_ratio",
            "rhythm_stream_visible_units",
            "rhythm_stream_full_units",
            "rhythm_reference_is_self",
            "rhythm_pair_is_identity",
            "ref_phrase_valid",
            "ref_phrase_boundary_strength",
        } or "stats" in key or "budget" in key:
            return torch.tensor(value, dtype=torch.float32)
        if key in {"sealed_mask", "boundary_confidence", "rhythm_offline_sealed_mask", "rhythm_offline_boundary_confidence"}:
            return torch.tensor(value, dtype=torch.float32)
        if key in {
            "rhythm_target_confidence",
            "rhythm_guidance_confidence",
            "rhythm_teacher_confidence",
            "rhythm_teacher_confidence_exec",
            "rhythm_teacher_confidence_budget",
            "rhythm_teacher_confidence_prefix",
            "rhythm_teacher_confidence_allocation",
            "rhythm_teacher_confidence_shape",
            "rhythm_offline_teacher_confidence",
            "dur_anchor_src",
            "prompt_duration_obs",
            "prompt_unit_mask",
            "unit_duration_tgt",
            "unit_confidence_tgt",
            "unit_confidence_local_tgt",
            "unit_confidence_coarse_tgt",
            "unit_alignment_coverage_tgt",
            "unit_alignment_match_tgt",
            "unit_alignment_cost_tgt",
        }:
            return torch.tensor(value, dtype=torch.float32)
        if key in {"rhythm_retimed_target_confidence", "rhythm_trace_horizon", "rhythm_source_phrase_threshold"}:
            return torch.tensor(value, dtype=torch.float32)
        if key in {
            "prompt_content_units",
            "phrase_group_index",
            "ref_phrase_lengths",
            "ref_phrase_starts",
            "ref_phrase_ends",
            "rhythm_offline_content_units",
            "rhythm_offline_dur_anchor_src",
            "rhythm_offline_open_run_mask",
            "rhythm_offline_sep_hint",
            "rhythm_offline_phrase_group_index",
            "selector_meta_indices",
            "selector_meta_starts",
            "selector_meta_ends",
            "rhythm_cache_version",
            "rhythm_unit_hop_ms",
            "rhythm_trace_hop_ms",
            "rhythm_trace_bins",
            "rhythm_slow_topk",
            "rhythm_selector_cell_size",
            "rhythm_reference_mode_id",
            "rhythm_teacher_target_source_id",
            "rhythm_retimed_target_source_id",
            "rhythm_pair_group_id",
            "rhythm_pair_rank",
        }:
            return torch.tensor(value, dtype=torch.long)
        if key == "rhythm_retimed_mel_tgt":
            return torch.tensor(value, dtype=torch.float32)
        if key == "rhythm_retimed_mel_len":
            return torch.tensor(value, dtype=torch.long)
        if key == "rhythm_retimed_frame_weight":
            return torch.tensor(value, dtype=torch.float32)
        return torch.tensor(value)

    def assemble(self, *, sample, item, ref_item, item_name: str):
        visible_len = int(sample["mel"].shape[0])
        full_content = self.owner._coerce_content_sequence(item["hubert"])
        full_content_len = len(full_content)
        if bool(self.hparams.get("rhythm_enable_v2", False)) and bool(
            self.hparams.get("rhythm_strict_content_mel_contract", True)
        ):
            tolerance = int(self.hparams.get("rhythm_content_mel_tolerance", 0) or 0)
            if abs(full_content_len - visible_len) > tolerance:
                raise RuntimeError(
                    f"Rhythm content/mel contract violated for {item_name}: hubert_len={full_content_len}, "
                    f"mel_len={visible_len}, tolerance={tolerance}. Re-binarize or align upstream hop settings."
                )
        if full_content_len < visible_len:
            raise RuntimeError(
                f"Rhythm content sequence shorter than mel for {item_name}: hubert_len={full_content_len}, "
                f"mel_len={visible_len}."
            )
        content_visible = full_content[:visible_len]
        if len(content_visible) != visible_len:
            raise RuntimeError(
                f"Visible content prefix mismatch for {item_name}: visible_tokens={len(content_visible)}, "
                f"mel_len={visible_len}."
            )
        sample["content"] = torch.LongTensor(content_visible)
        target_mode = self.owner._resolve_rhythm_target_mode()
        full_visible_tokens = np.asarray(content_visible, dtype=np.int64)
        stream_visible_tokens = self.owner._select_streaming_visible_tokens(
            full_visible_tokens,
            item_name=item_name,
        )

        optional_rhythm_keys = self.owner._resolve_optional_sample_keys()
        rhythm_runtime_fields = {}
        target_source_cache = self.owner._get_source_rhythm_cache(item, stream_visible_tokens, target_mode=target_mode)
        maybe_build_duration_v3_training_source_cache = getattr(
            self.owner,
            "_maybe_build_duration_v3_training_source_cache",
            None,
        )
        if callable(maybe_build_duration_v3_training_source_cache):
            source_cache = maybe_build_duration_v3_training_source_cache(
                target_source_cache,
                item_name=item_name,
            )
        else:
            source_cache = target_source_cache
        stream_units = int(np.asarray(source_cache["dur_anchor_src"]).reshape(-1).shape[0])
        offline_units = stream_units
        if int(stream_visible_tokens.shape[0]) < int(full_visible_tokens.shape[0]):
            if self.owner._should_export_streaming_offline_sidecars():
                offline_source_cache = self.owner._get_source_rhythm_cache(item, full_visible_tokens, target_mode=target_mode)
                rhythm_runtime_fields.update(self.owner._prefix_source_cache(offline_source_cache, prefix="rhythm_offline_"))
                offline_units = int(np.asarray(offline_source_cache["dur_anchor_src"]).reshape(-1).shape[0])
                if self.owner._should_export_offline_teacher_aux():
                    rhythm_runtime_fields.update(
                        self.owner._build_offline_teacher_aux_fields(
                            item,
                            offline_units=offline_units,
                        )
                    )
            else:
                offline_units = int(
                    np.asarray(
                        self.owner._get_source_rhythm_cache(item, full_visible_tokens, target_mode=target_mode)["dur_anchor_src"]
                    ).reshape(-1).shape[0]
                )
        if self.owner._should_export_streaming_prefix_meta():
            rhythm_runtime_fields["rhythm_stream_visible_units"] = np.asarray([stream_units], dtype=np.float32)
            rhythm_runtime_fields["rhythm_stream_full_units"] = np.asarray([offline_units], dtype=np.float32)
            rhythm_runtime_fields["rhythm_stream_prefix_ratio"] = np.asarray(
                [float(stream_units) / float(max(offline_units, 1))],
                dtype=np.float32,
            )
        rhythm_ref_item = item if self.owner._should_use_self_rhythm_reference(item, target_mode=target_mode) else ref_item
        reference_is_self = bool(rhythm_ref_item is item or rhythm_ref_item is None)
        identity_flag = sample.get("rhythm_pair_is_identity", 0.0)
        if isinstance(identity_flag, torch.Tensor):
            explicit_identity_pair = bool(
                sample.get("_rhythm_pair_manifest_entry", False)
                and float(identity_flag.reshape(-1)[0].item()) > 0.5
            )
        else:
            explicit_identity_pair = bool(
                sample.get("_rhythm_pair_manifest_entry", False)
                and float(identity_flag) > 0.5
            )
        allow_identity_pair = bool(self.hparams.get("rhythm_allow_identity_pairs", False))
        if (
            reference_is_self
            and bool(self.hparams.get("rhythm_require_external_reference", False))
            and not (explicit_identity_pair and allow_identity_pair)
        ):
            raise RuntimeError(
                f"Rhythm stage requires an external reference, but {item_name} resolved to self-conditioning. "
                "This usually means the speaker pool collapsed to a singleton after filtering. "
                "Relax rhythm_require_external_reference, enable rhythm_allow_identity_pairs for explicit A|A anchors, "
                "or rebuild the split with at least two items per speaker."
            )
        rhythm_runtime_fields["rhythm_reference_is_self"] = np.asarray(
            [1.0 if reference_is_self else 0.0],
            dtype=np.float32,
        )
        if "rhythm_pair_group_id" in sample:
            rhythm_runtime_fields["rhythm_pair_group_id"] = np.asarray(
                sample["rhythm_pair_group_id"].cpu().numpy(),
                dtype=np.int64,
            )
        if "rhythm_pair_rank" in sample:
            rhythm_runtime_fields["rhythm_pair_rank"] = np.asarray(
                sample["rhythm_pair_rank"].cpu().numpy(),
                dtype=np.int64,
            )
        if "rhythm_pair_is_identity" in sample:
            rhythm_runtime_fields["rhythm_pair_is_identity"] = np.asarray(
                sample["rhythm_pair_is_identity"].cpu().numpy(),
                dtype=np.float32,
            )
        resolve_paired_target_item = getattr(self.owner, "_resolve_paired_target_rhythm_item", None)
        paired_target_item = None
        if callable(resolve_paired_target_item):
            paired_target_item = resolve_paired_target_item(
                sample=sample,
                item=item,
                target_mode=target_mode,
            )
        ref_conditioning = self._build_reference_conditioning(
            rhythm_ref_item=rhythm_ref_item,
            sample=sample,
            item=item,
            target_mode=target_mode,
        )
        build_paired_target_conditioning = getattr(self.owner, "_build_paired_target_rhythm_conditioning", None)
        paired_target_conditioning = {}
        if callable(build_paired_target_conditioning):
            paired_target_conditioning = build_paired_target_conditioning(
                paired_target_item,
                sample,
                target_mode=target_mode,
                item=item,
            )
        rhythm_runtime_fields.update(source_cache)
        rhythm_runtime_fields.update(ref_conditioning)
        try:
            merged_targets = self.owner._merge_rhythm_targets(
                item,
                target_source_cache,
                ref_conditioning,
                paired_target_conditioning,
                sample,
            )
        except TypeError as exc:
            if "positional arguments" not in str(exc):
                raise
            merged_targets = self.owner._merge_rhythm_targets(
                item,
                target_source_cache,
                ref_conditioning,
                sample,
            )
        rhythm_runtime_fields.update(merged_targets)

        for key in optional_rhythm_keys:
            if key in rhythm_runtime_fields:
                value = rhythm_runtime_fields[key]
            elif key in item:
                value = item[key]
            else:
                continue
            sample[key] = self._tensorize_optional_value(key, value)
        sample.pop("ref_item_id", None)
        sample.pop("paired_target_item_id", None)
        return sample
