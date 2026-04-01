
from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
import numpy as np
from modules.Conan.rhythm.supervision import (
    build_reference_guided_targets,
    build_reference_rhythm_conditioning,
    build_reference_teacher_targets,
    build_source_rhythm_cache,
)

class ConanDataset(FastSpeechDataset):
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

    def _get_source_rhythm_cache(self, item, visible_tokens):
        cache_keys = ("content_units", "dur_anchor_src", "open_run_mask", "sep_hint")
        full_tokens = np.asarray(item["hubert"])
        visible_tokens = np.asarray(visible_tokens)
        if all(key in item for key in cache_keys) and int(full_tokens.shape[0]) == int(visible_tokens.shape[0]):
            return {key: item[key] for key in cache_keys}
        return build_source_rhythm_cache(
            visible_tokens,
            silent_token=self.hparams.get("silent_token", 57),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
        )

    def _get_reference_rhythm_conditioning(self, ref_item, sample):
        cache_keys = ("ref_rhythm_stats", "ref_rhythm_trace")
        if (
            ref_item is not None
            and all(key in ref_item for key in cache_keys)
            and "mel" in ref_item
            and int(sample["ref_mel"].shape[0]) == int(np.asarray(ref_item["mel"]).shape[0])
        ):
            return {key: ref_item[key] for key in cache_keys}
        return build_reference_rhythm_conditioning(
            sample["ref_mel"],
            trace_bins=int(self.hparams.get("rhythm_trace_bins", 24)),
        )

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

    def _merge_rhythm_targets(self, item, source_cache, ref_conditioning):
        target_mode = self._resolve_rhythm_target_mode()
        cached_targets = {key: item[key] for key in self._RHYTHM_TARGET_KEYS if key in item}
        if target_mode == "cached_only":
            return cached_targets

        runtime_targets = self._build_runtime_rhythm_targets(source_cache, ref_conditioning)
        if target_mode == "runtime_only":
            return runtime_targets

        merged = dict(cached_targets)
        for key, value in runtime_targets.items():
            merged.setdefault(key, value)
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

        optional_rhythm_keys = [
            "content_units",
            "dur_anchor_src",
            "open_run_mask",
            "sep_hint",
            "ref_rhythm_stats",
            "ref_rhythm_trace",
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
        source_cache = self._get_source_rhythm_cache(item, sample["content"].cpu().numpy())
        ref_conditioning = self._get_reference_rhythm_conditioning(ref_item, sample)
        rhythm_runtime_fields.update(source_cache)
        rhythm_runtime_fields.update(ref_conditioning)
        rhythm_runtime_fields.update(
            self._merge_rhythm_targets(
                item,
                source_cache,
                ref_conditioning,
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
            "sep_hint": ("long", 0),
            "ref_rhythm_stats": ("float", 0.0),
            "ref_rhythm_trace": ("float", 0.0),
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
