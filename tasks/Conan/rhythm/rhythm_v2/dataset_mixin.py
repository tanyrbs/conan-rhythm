from __future__ import annotations

import numpy as np

from modules.Conan.rhythm.supervision import build_reference_rhythm_conditioning


class RhythmV2DatasetMixin:
    @staticmethod
    def _build_offline_teacher_aux_fields(item, *, offline_units: int) -> dict:
        speech_key = "rhythm_teacher_speech_exec_tgt"
        pause_key = "rhythm_teacher_pause_exec_tgt" if "rhythm_teacher_pause_exec_tgt" in item else "rhythm_teacher_blank_exec_tgt"
        if speech_key not in item or pause_key not in item:
            return {}
        speech = np.asarray(item[speech_key]).reshape(-1).astype(np.float32)
        pause = np.asarray(item[pause_key]).reshape(-1).astype(np.float32)
        teacher_units = min(int(offline_units), int(speech.shape[0]), int(pause.shape[0]))
        if teacher_units <= 0:
            return {}
        speech = speech[:teacher_units]
        pause = pause[:teacher_units]
        fields = {
            "rhythm_offline_teacher_speech_exec_tgt": speech,
            "rhythm_offline_teacher_pause_exec_tgt": pause,
            "rhythm_offline_teacher_speech_budget_tgt": np.asarray([float(speech.sum())], dtype=np.float32),
            "rhythm_offline_teacher_pause_budget_tgt": np.asarray([float(pause.sum())], dtype=np.float32),
        }
        if "rhythm_teacher_confidence" in item:
            fields["rhythm_offline_teacher_confidence"] = np.asarray(item["rhythm_teacher_confidence"]).reshape(-1)[:1].astype(np.float32)
        return fields

    def _build_runtime_rhythm_targets(self, source_cache, ref_conditioning):
        return self._rhythm_target_builder().build_runtime_rhythm_targets(source_cache, ref_conditioning)

    def _build_legacy_reference_rhythm_conditioning(
        self,
        ref_item,
        sample,
        *,
        target_mode: str,
        prompt_conditioning: dict | None = None,
    ):
        cache_keys = self._RHYTHM_REF_CACHE_KEYS
        prompt_conditioning = prompt_conditioning or {}
        if ref_item is not None and all(key in ref_item for key in cache_keys):
            conditioning = {key: ref_item[key] for key in cache_keys}
            conditioning.update(prompt_conditioning)
            for debug_key in (
                self._RHYTHM_REF_DEBUG_CACHE_KEYS
                + self._RHYTHM_REF_PLANNER_DEBUG_CACHE_KEYS
                + self._RHYTHM_REF_PHRASE_CACHE_KEYS
            ):
                if debug_key in ref_item:
                    conditioning[debug_key] = ref_item[debug_key]
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
            smooth_kernel=int(self.hparams.get("rhythm_trace_smooth_kernel", 5)),
            slow_topk=int(self.hparams.get("rhythm_slow_topk", 6)),
            selector_cell_size=int(self.hparams.get("rhythm_selector_cell_size", 3)),
        )
        conditioning.update(prompt_conditioning)
        self._validate_reference_conditioning_shapes(conditioning, item_name="<runtime-ref-conditioning>")
        return conditioning

    def _merge_legacy_rhythm_targets(self, item, source_cache, ref_conditioning, sample):
        return self._rhythm_target_builder().merge_rhythm_targets(
            item=item,
            source_cache=source_cache,
            ref_conditioning=ref_conditioning,
            sample=sample,
        )


__all__ = ["RhythmV2DatasetMixin"]
