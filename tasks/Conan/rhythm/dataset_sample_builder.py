from __future__ import annotations

import numpy as np
import torch


class RhythmDatasetSampleAssembler:
    def __init__(self, owner):
        self.owner = owner

    @property
    def hparams(self):
        return self.owner.hparams

    def _tensorize_optional_value(self, key, value):
        if key in {"ref_rhythm_trace", "slow_rhythm_memory"}:
            return torch.tensor(value, dtype=torch.float32)
        if key in {
            "source_boundary_cue",
            "phrase_group_pos",
            "phrase_final_mask",
            "slow_rhythm_summary",
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
        }:
            return torch.tensor(value, dtype=torch.float32)
        if key in {"rhythm_retimed_target_confidence", "rhythm_trace_horizon", "rhythm_source_phrase_threshold"}:
            return torch.tensor(value, dtype=torch.float32)
        if key in {
            "phrase_group_index",
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
        source_cache = self.owner._get_source_rhythm_cache(item, stream_visible_tokens, target_mode=target_mode)
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
        ref_conditioning = self.owner._get_reference_rhythm_conditioning(rhythm_ref_item, sample, target_mode=target_mode)
        rhythm_runtime_fields.update(source_cache)
        rhythm_runtime_fields.update(ref_conditioning)
        rhythm_runtime_fields.update(
            self.owner._merge_rhythm_targets(
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
            sample[key] = self._tensorize_optional_value(key, value)
        sample.pop("ref_item_id", None)
        return sample
