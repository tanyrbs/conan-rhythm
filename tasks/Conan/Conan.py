from modules.Conan.Conan import Conan, ConanPostnet
from tasks.Conan.base_gen_task import AuxDecoderMIDITask, f0_to_figure
from utils.commons.hparams import hparams
import torch
from utils.commons.ckpt_utils import load_ckpt
from tasks.Conan.dataset import ConanDataset
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
import math
from modules.tts.iclspeech.multi_window_disc import Discriminator
import torch.nn as nn
import random
from tasks.Conan.rhythm.losses import RhythmLossTargets, build_rhythm_loss_dict
from tasks.Conan.rhythm.metrics import build_rhythm_metric_dict, build_streaming_chunk_metrics
from tasks.Conan.rhythm.streaming_eval import run_chunkwise_streaming_inference
from modules.Conan.rhythm.bridge import resolve_rhythm_apply_mode
from utils.nn.seq_utils import weights_nonzero_speech
from utils.metrics.ssim import ssim


class ConanEmbTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def run_model(self, sample):
        with torch.no_grad():
            ref = sample['mels']
            output = self.model.encode_spk_embed(ref.transpose(1, 2)).squeeze(2)
        return {}, {"style_embed": output}


class ConanTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        self._warned_retimed_pitch_supervision = False
        self._disc_skip_for_retimed = False
        self._validate_rhythm_training_hparams()
        self.build_disc_model()

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        if bool(hparams.get("rhythm_optimize_module_only", False)):
            rhythm_params = self._collect_rhythm_gen_params()
            self.gen_params = rhythm_params if len(rhythm_params) > 0 else [p for p in self.model.parameters() if p.requires_grad]
        else:
            self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    @staticmethod
    def _validate_rhythm_training_hparams():
        if not bool(hparams.get("rhythm_enable_v2", False)):
            return
        errors = []
        warnings = []
        target_mode = ConanTask._resolve_rhythm_target_mode()
        schedule_only = bool(hparams.get("rhythm_schedule_only_stage", False))
        distill_surface = str(hparams.get("rhythm_distill_surface", "auto") or "auto").strip().lower()
        lambda_distill = float(hparams.get("lambda_rhythm_distill", 0.0))
        lambda_guidance = float(hparams.get("lambda_rhythm_guidance", 0.0))
        distill_budget_weight = float(hparams.get("rhythm_distill_budget_weight", 0.5))
        distill_allocation_weight = float(hparams.get("rhythm_distill_allocation_weight", 0.5))
        distill_prefix_weight = float(hparams.get("rhythm_distill_prefix_weight", 0.25))
        lambda_carry = hparams.get("lambda_rhythm_carry", None)
        lambda_cumplan = hparams.get("lambda_rhythm_cumplan", None)
        enable_dual_teacher = bool(hparams.get("rhythm_enable_dual_mode_teacher", False))
        enable_learned_offline_teacher = bool(hparams.get("rhythm_enable_learned_offline_teacher", True))
        retimed_target_mode = ConanTask._resolve_retimed_target_mode()
        use_retimed_pitch_target = bool(hparams.get("rhythm_use_retimed_pitch_target", False))
        disable_pitch_when_retimed = bool(hparams.get("rhythm_disable_pitch_loss_when_retimed", True))
        if lambda_carry is not None and lambda_cumplan is not None:
            if abs(float(lambda_carry) - float(lambda_cumplan)) > 1e-8:
                errors.append("lambda_rhythm_carry and lambda_rhythm_cumplan are both set but disagree.")
            else:
                warnings.append("Both lambda_rhythm_carry and lambda_rhythm_cumplan are set; prefer lambda_rhythm_cumplan.")
        if lambda_distill > 0.0:
            if distill_surface in {"none", "off", "disable", "disabled", "false"}:
                errors.append("lambda_rhythm_distill > 0 but rhythm_distill_surface disables distillation.")
            if distill_surface in {"offline", "full_context", "shared_offline"} and not bool(
                hparams.get("rhythm_enable_dual_mode_teacher", False)
            ):
                errors.append("Offline distillation requires rhythm_enable_dual_mode_teacher: true.")
        if enable_dual_teacher and not enable_learned_offline_teacher:
            errors.append("rhythm_enable_dual_mode_teacher requires rhythm_enable_learned_offline_teacher: true.")
        if enable_dual_teacher and lambda_distill <= 0.0:
            warnings.append("Dual-mode teacher is enabled but lambda_rhythm_distill == 0.")
        distill_conf_floor = float(hparams.get("rhythm_distill_confidence_floor", 0.05))
        distill_conf_power = float(hparams.get("rhythm_distill_confidence_power", 1.0))
        source_boundary_scale = float(hparams.get("rhythm_source_boundary_scale", 1.0))
        source_boundary_scale_train_start = float(hparams.get("rhythm_source_boundary_scale_train_start", 1.0))
        source_boundary_scale_train_end = float(hparams.get("rhythm_source_boundary_scale_train_end", source_boundary_scale))
        source_boundary_scale_anneal_steps = int(hparams.get("rhythm_source_boundary_scale_anneal_steps", 20000) or 0)
        source_boundary_scale_warmup_steps = int(hparams.get("rhythm_source_boundary_scale_warmup_steps", 0) or 0)
        teacher_source_boundary_scale = float(
            hparams.get("rhythm_teacher_source_boundary_scale", source_boundary_scale)
        )
        if not (0.0 < distill_conf_floor <= 1.0):
            errors.append("rhythm_distill_confidence_floor must be in (0, 1].")
        if distill_conf_power <= 0.0:
            errors.append("rhythm_distill_confidence_power must be > 0.")
        if source_boundary_scale < 0.0 or source_boundary_scale_train_start < 0.0 or source_boundary_scale_train_end < 0.0:
            errors.append("rhythm_source_boundary_scale* must be >= 0.")
        if teacher_source_boundary_scale < 0.0:
            errors.append("rhythm_teacher_source_boundary_scale must be >= 0.")
        if source_boundary_scale_anneal_steps < 0 or source_boundary_scale_warmup_steps < 0:
            errors.append("rhythm_source_boundary_scale warmup/anneal steps must be >= 0.")
        if source_boundary_scale_train_start < source_boundary_scale_train_end:
            warnings.append("Source-boundary prior is configured weak->strong over training; maintained path usually anneals strong->soft.")
        if source_boundary_scale > 1.0 or teacher_source_boundary_scale > 1.0:
            warnings.append("Source-boundary prior scale > 1.0 increases source phrasing lock-in risk.")
        if schedule_only and bool(hparams.get("rhythm_apply_train_override", False)):
            errors.append("rhythm_schedule_only_stage should not enable train-time retimed rendering.")
        if bool(hparams.get("rhythm_apply_train_override", False)) and not bool(
            hparams.get("rhythm_use_retimed_target_if_available", False)
        ):
            errors.append("Train-time retimed rendering requires rhythm_use_retimed_target_if_available: true.")
        if bool(hparams.get("rhythm_apply_train_override", False)):
            retimed_source = str(hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower()
            primary_surface = str(hparams.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()
            if retimed_source == "teacher" and primary_surface != "teacher":
                errors.append("Teacher retimed targets should pair with rhythm_primary_target_surface: teacher.")
            if retimed_target_mode == "online" and bool(hparams.get("rhythm_require_retimed_cache", False)):
                warnings.append("Online retimed target mode keeps rhythm_require_retimed_cache enabled; cached targets will be treated only as a safety fallback.")
            if not use_retimed_pitch_target and not disable_pitch_when_retimed:
                errors.append(
                    "Train-time retimed rendering must either enable rhythm_use_retimed_pitch_target "
                    "or set rhythm_disable_pitch_loss_when_retimed: true."
                )
            if use_retimed_pitch_target:
                warnings.append("Train-time retimed rendering will remap F0/UV onto the online frame plan before pitch supervision.")
            else:
                warnings.append("Train-time retimed rendering disables source-aligned pitch supervision unless retimed pitch targets exist.")
        if str(hparams.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower() == "teacher":
            if not bool(hparams.get("rhythm_binarize_teacher_targets", False)):
                warnings.append("Primary rhythm target surface is teacher but rhythm_binarize_teacher_targets is false.")
        if str(hparams.get("rhythm_binarize_retimed_mel_source", "guidance") or "guidance").strip().lower() == "teacher":
            if not bool(hparams.get("rhythm_binarize_teacher_targets", False)):
                errors.append("Teacher retimed targets require rhythm_binarize_teacher_targets: true.")
        if (
            target_mode == "cached_only"
            and bool(hparams.get("rhythm_require_cached_teacher", False))
            and bool(hparams.get("rhythm_dataset_build_teacher_from_ref", False))
        ):
            warnings.append(
                "cached_only + require_cached_teacher is active, but rhythm_dataset_build_teacher_from_ref=true "
                "still leaves a runtime fallback path enabled."
            )
        if (
            target_mode == "cached_only"
            and str(hparams.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower() == "teacher"
            and lambda_guidance <= 0.0
            and bool(hparams.get("rhythm_dataset_build_guidance_from_ref", True))
        ):
            warnings.append(
                "Teacher-first cached training still builds guidance-from-ref even though lambda_rhythm_guidance == 0. "
                "Disable rhythm_dataset_build_guidance_from_ref to narrow the formal training path."
            )
        if lambda_guidance > 0.0 and schedule_only:
            warnings.append("lambda_rhythm_guidance > 0 during schedule-only stage increases migration-path complexity.")
        if enable_dual_teacher or bool(hparams.get("rhythm_apply_train_override", False)):
            if lambda_guidance > 0.0:
                warnings.append("Maintained dual-mode/retimed path recommends lambda_rhythm_guidance: 0.")
            if distill_allocation_weight > 0.0:
                warnings.append("Maintained dual-mode/retimed path recommends rhythm_distill_allocation_weight: 0.")
            if distill_budget_weight > 0.15:
                warnings.append("Maintained dual-mode/retimed path recommends a tiny rhythm_distill_budget_weight (<= 0.15).")
            if distill_prefix_weight <= 0.0 and lambda_distill > 0.0:
                warnings.append("Dual-mode distillation without prefix supervision is weaker than the maintained path.")
        if "rhythm_min_unit_frames" in hparams:
            errors.append(
                "rhythm_min_unit_frames has been removed from the maintained rhythm path; delete the key or implement it explicitly before training."
            )
        if errors:
            raise ValueError("Invalid Rhythm V2 training config:\n- " + "\n- ".join(errors))
        if warnings:
            print("| Rhythm V2 config warnings:")
            for warning in warnings:
                print(f"|   - {warning}")

    def _collect_rhythm_gen_params(self):
        if self.model is None or not getattr(self.model, "rhythm_enable_v2", False):
            return []
        params = []
        if getattr(self.model, "rhythm_module", None) is not None:
            params.extend(list(self.model.rhythm_module.parameters()))
        if bool(hparams.get("rhythm_optimize_pause_state", False)) and getattr(self.model, "rhythm_pause_state", None) is not None:
            params.append(self.model.rhythm_pause_state)
        if getattr(self.model, "rhythm_render_phase_mlp", None) is not None:
            params.extend(list(self.model.rhythm_render_phase_mlp.parameters()))
        if getattr(self.model, "rhythm_render_phase_gain", None) is not None:
            params.append(self.model.rhythm_render_phase_gain)
        dedup = []
        seen = set()
        for param in params:
            if param is None or not getattr(param, "requires_grad", False):
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(param)
        return dedup

    @staticmethod
    def _get_rhythm_cumplan_lambda() -> float:
        if "lambda_rhythm_cumplan" in hparams:
            return float(hparams.get("lambda_rhythm_cumplan", 0.15))
        return float(hparams.get("lambda_rhythm_carry", 0.15))

    @staticmethod
    def _resolve_rhythm_pause_boundary_weight() -> float:
        # Backward compatible alias:
        # `rhythm_pause_exec_boundary_boost` (newer) falls back to
        # `rhythm_pause_boundary_weight` (older).
        if "rhythm_pause_exec_boundary_boost" in hparams:
            return float(hparams.get("rhythm_pause_exec_boundary_boost", 0.75))
        return float(hparams.get("rhythm_pause_boundary_weight", 0.35))

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    @staticmethod
    def _parse_optional_bool(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"", "none", "null", "auto", "default"}:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Unsupported optional bool value: {value}")

    def _resolve_rhythm_apply_override(self, *, infer: bool, test: bool, explicit=None, current_step=None):
        explicit_value = self._parse_optional_bool(explicit)
        if explicit_value is not None:
            enabled = explicit_value
        else:
            stage = "test" if test else ("valid" if infer else "train")
            enabled = self._parse_optional_bool(hparams.get(f"rhythm_apply_{stage}_override", None))
        if enabled is None:
            return None
        effective_step = int(self.global_step if current_step is None else current_step)
        stage = "test" if test else ("valid" if infer else "train")
        start_step = int(hparams.get(f"rhythm_{stage}_render_start_steps", 0) or 0)
        if enabled and stage in {"train", "valid"} and effective_step < start_step:
            return False
        retimed_target_start = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
        if (
            enabled
            and stage in {"train", "valid"}
            and bool(hparams.get("rhythm_use_retimed_target_if_available", False))
            and effective_step < retimed_target_start
        ):
            return False
        return enabled

    @staticmethod
    def _resolve_rhythm_target_mode() -> str:
        mode = str(hparams.get("rhythm_dataset_target_mode", "prefer_cache") or "prefer_cache").strip().lower()
        aliases = {
            "auto": "prefer_cache",
            "offline": "cached_only",
            "offline_only": "cached_only",
            "never": "cached_only",
            "runtime": "runtime_only",
            "always": "runtime_only",
        }
        return aliases.get(mode, mode)

    @staticmethod
    def _resolve_rhythm_primary_target_surface() -> str:
        surface = str(hparams.get("rhythm_primary_target_surface", "guidance") or "guidance").strip().lower()
        aliases = {
            "cache_teacher": "teacher",
            "offline": "teacher",
            "offline_teacher": "teacher",
            "teacher_surface": "teacher",
            "guidance_surface": "guidance",
            "self": "guidance",
        }
        resolved = aliases.get(surface, surface)
        if resolved not in {"guidance", "teacher"}:
            raise ValueError(f"Unsupported rhythm_primary_target_surface: {surface}")
        return resolved

    @staticmethod
    def _resolve_rhythm_distill_surface() -> str:
        surface = str(hparams.get("rhythm_distill_surface", "auto") or "auto").strip().lower()
        aliases = {
            "off": "none",
            "disable": "none",
            "disabled": "none",
            "false": "none",
            "cache_teacher": "cache",
            "cached_teacher": "cache",
            "full_context": "offline",
            "shared_offline": "offline",
            "algo": "algorithmic",
            "teacher": "cache",
        }
        resolved = aliases.get(surface, surface)
        if resolved not in {"auto", "none", "cache", "offline", "algorithmic"}:
            raise ValueError(f"Unsupported rhythm_distill_surface: {surface}")
        return resolved

    @staticmethod
    def _collect_rhythm_source_cache(sample, *, prefix: str = ""):
        keys = (
            "content_units",
            "dur_anchor_src",
            "open_run_mask",
            "sealed_mask",
            "sep_hint",
            "boundary_confidence",
        )
        payload = {}
        for key in keys:
            sample_key = f"{prefix}{key}"
            value = sample.get(sample_key)
            if value is not None:
                payload[key] = value
        return payload or None

    @staticmethod
    def _slice_rhythm_surface_to_student(
        *,
        speech_exec,
        pause_exec,
        student_units: int,
        dur_anchor_src: torch.Tensor,
        unit_mask: torch.Tensor,
    ):
        speech_exec = speech_exec[:, :student_units]
        pause_exec = pause_exec[:, :student_units]
        speech_budget = speech_exec.float().sum(dim=1, keepdim=True)
        pause_budget = pause_exec.float().sum(dim=1, keepdim=True)
        allocation = (speech_exec.float() + pause_exec.float()) * unit_mask[:, :student_units].float()
        prefix_clock, prefix_backlog = ConanTask._build_prefix_carry_from_exec(
            speech_exec,
            pause_exec,
            dur_anchor_src[:, :student_units],
            unit_mask[:, :student_units],
        )
        return (
            speech_exec,
            pause_exec,
            speech_budget,
            pause_budget,
            allocation,
            prefix_clock,
            prefix_backlog,
        )

    @staticmethod
    def _build_prefix_carry_from_exec(speech_exec, pause_exec, dur_anchor_src, unit_mask):
        unit_mask = unit_mask.float()
        prefix_clock = torch.cumsum(
            ((speech_exec.float() + pause_exec.float()) - dur_anchor_src.float()) * unit_mask,
            dim=1,
        ) * unit_mask
        prefix_backlog = prefix_clock.clamp_min(0.0) * unit_mask
        return prefix_clock, prefix_backlog

    @staticmethod
    def _normalize_distill_confidence(distill_confidence, *, batch_size: int, device: torch.device):
        floor = float(hparams.get("rhythm_distill_confidence_floor", 0.05))
        power = float(hparams.get("rhythm_distill_confidence_power", 1.0))
        if distill_confidence is None:
            confidence = torch.ones((batch_size, 1), device=device)
        else:
            confidence = distill_confidence.float().reshape(batch_size, -1)[:, :1].to(device=device)
        confidence = confidence.clamp(min=floor, max=1.0)
        if abs(power - 1.0) > 1e-8:
            confidence = confidence.pow(power)
        return confidence

    @staticmethod
    def _normalize_component_distill_confidence(
        component_confidence,
        *,
        fallback_confidence: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ):
        if component_confidence is None:
            return fallback_confidence
        return ConanTask._normalize_distill_confidence(
            component_confidence,
            batch_size=batch_size,
            device=device,
        )

    @staticmethod
    def _resolve_retimed_target_mode() -> str:
        mode = str(hparams.get("rhythm_retimed_target_mode", "cached") or "cached").strip().lower()
        aliases = {
            "cache": "cached",
            "cached_only": "cached",
            "teacher": "cached",
            "runtime": "online",
            "online_only": "online",
            "mixed": "hybrid",
        }
        resolved = aliases.get(mode, mode)
        if resolved not in {"cached", "online", "hybrid"}:
            raise ValueError(f"Unsupported rhythm_retimed_target_mode: {mode}")
        return resolved

    @staticmethod
    def _merge_retimed_weight(frame_weight, confidence):
        if frame_weight is None and confidence is None:
            return None
        if confidence is not None:
            confidence = confidence.float().clamp_min(float(hparams.get("rhythm_retimed_confidence_floor", 0.05)))
        if frame_weight is None:
            return confidence
        frame_weight = frame_weight.float()
        if confidence is None:
            return frame_weight
        while confidence.dim() < frame_weight.dim():
            confidence = confidence.unsqueeze(-1)
        return frame_weight * confidence

    def _resolve_acoustic_target_post_model(
        self,
        sample,
        model_out,
        *,
        apply_rhythm_render: bool,
        infer: bool,
        test: bool,
        current_step=None,
    ):
        target = sample["mels"]
        frame_weight = None
        is_retimed = False
        source = "source"
        effective_step = int(self.global_step if current_step is None else current_step)
        start_step = int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
        if (
            not bool(apply_rhythm_render)
            or not bool(hparams.get("rhythm_use_retimed_target_if_available", False))
            or effective_step < start_step
        ):
            return target, is_retimed, frame_weight, source

        stage = "test" if test else ("valid" if infer else "train")
        target_mode = self._resolve_retimed_target_mode()
        online_start = int(hparams.get("rhythm_online_retimed_target_start_steps", start_step) or start_step)
        online_ready = effective_step >= online_start
        prefer_online = target_mode in {"online", "hybrid"} and online_ready

        if prefer_online:
            online_target = model_out.get("rhythm_online_retimed_mel_tgt")
            if online_target is not None:
                target = online_target
                frame_weight = self._merge_retimed_weight(
                    model_out.get("rhythm_online_retimed_frame_weight"),
                    None,
                )
                is_retimed = True
                source = "online"
                return target, is_retimed, frame_weight, source
            if target_mode == "online":
                raise RuntimeError(
                    f"Rhythm online retimed target is required for the active render path ({stage}) but is unavailable."
                )

        cached_target = sample.get("rhythm_retimed_mel_tgt")
        if cached_target is not None:
            target = cached_target
            frame_weight = self._merge_retimed_weight(
                sample.get("rhythm_retimed_frame_weight"),
                sample.get("rhythm_retimed_target_confidence"),
            )
            is_retimed = True
            source = "cached"
            return target, is_retimed, frame_weight, source

        require_retimed = bool(hparams.get("rhythm_require_retimed_cache", False))
        if require_retimed or (not test and self._resolve_rhythm_target_mode() == "cached_only"):
            raise RuntimeError(
                "Rhythm retimed target is required for the active render path "
                f"({stage}) but neither online nor cached retimed targets are available."
            )
        return target, is_retimed, frame_weight, source

    @staticmethod
    def _resample_sequence_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        if target_len <= 0:
            return x[:, :0]
        if x.size(1) <= 0:
            return x.new_zeros((x.size(0), target_len, x.size(2)))
        if x.dim() != 3:
            raise ValueError(f"Expected rank-3 tensor [B,T,C], got {tuple(x.shape)}")
        resized = F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False,
        )
        return resized.transpose(1, 2)

    @staticmethod
    def _resample_weight_length(weight: torch.Tensor, target_len: int) -> torch.Tensor:
        if weight.size(1) == target_len:
            return weight
        if target_len <= 0:
            return weight[:, :0]
        if weight.size(1) <= 0:
            return weight.new_zeros((weight.size(0), target_len))
        resized = F.interpolate(
            weight.unsqueeze(1),
            size=target_len,
            mode='linear',
            align_corners=False,
        )
        return resized.squeeze(1)

    def _align_acoustic_target_to_output(self, mel_out, acoustic_target, acoustic_weight):
        if mel_out.size(1) == acoustic_target.size(1):
            return mel_out, acoustic_target, acoustic_weight
        if bool(hparams.get("rhythm_resample_retimed_target_to_output", True)):
            acoustic_target = self._resample_sequence_length(acoustic_target, mel_out.size(1))
            if acoustic_weight is not None:
                acoustic_weight = self._resample_weight_length(acoustic_weight.float(), mel_out.size(1))
            return mel_out, acoustic_target, acoustic_weight
        target_len = min(int(mel_out.size(1)), int(acoustic_target.size(1)))
        acoustic_target = acoustic_target[:, :target_len]
        mel_out = mel_out[:, :target_len]
        if acoustic_weight is not None:
            acoustic_weight = acoustic_weight[:, :target_len]
        return mel_out, acoustic_target, acoustic_weight

    @staticmethod
    def _expand_frame_weight(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if weight is None:
            return weights_nonzero_speech(target)
        weight = weight.float()
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        while weight.dim() < target.dim():
            weight = weight.unsqueeze(-1)
        return weight

    def _weighted_l1_loss(self, decoder_output, target, frame_weight):
        loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self._expand_frame_weight(frame_weight, target)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _weighted_mse_loss(self, decoder_output, target, frame_weight):
        loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = self._expand_frame_weight(frame_weight, target)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _weighted_ssim_loss(self, decoder_output, target, frame_weight, bias=6.0):
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        weights = self._expand_frame_weight(frame_weight, target[:, 0])
        return (ssim_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _update_public_loss_aliases(self, losses):
        device = None
        for value in losses.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        zero = torch.tensor(0.0, device=device or "cpu")
        if "rhythm_exec_speech" in losses:
            losses["L_exec_speech"] = losses["rhythm_exec_speech"].detach()
        if "rhythm_exec_pause" in losses:
            losses["L_exec_pause"] = losses["rhythm_exec_pause"].detach()
        losses["L_budget"] = losses.get("rhythm_budget", zero).detach() if isinstance(losses.get("rhythm_budget"), torch.Tensor) else zero
        cumplan = losses.get("rhythm_cumplan", losses.get("rhythm_carry"))
        losses["L_cumplan"] = cumplan.detach() if isinstance(cumplan, torch.Tensor) else zero
        losses["L_prefix_state"] = losses["L_cumplan"]
        losses["L_kd"] = losses.get("rhythm_distill", zero).detach() if isinstance(losses.get("rhythm_distill"), torch.Tensor) else zero
        rhythm_exec = losses.get("rhythm_exec")
        losses["L_rhythm_exec"] = rhythm_exec.detach() if isinstance(rhythm_exec, torch.Tensor) else zero
        stream_state = losses.get("rhythm_stream_state")
        losses["L_stream_state"] = stream_state.detach() if isinstance(stream_state, torch.Tensor) else zero
        pitch_value = losses.get("pitch")
        if not isinstance(pitch_value, torch.Tensor):
            pitch_value = zero
            for loss_name in ("fdiff", "uv", "pflow", "gdiff", "mdiff"):
                value = losses.get(loss_name)
                if isinstance(value, torch.Tensor):
                    pitch_value = pitch_value + value.detach()
        else:
            pitch_value = pitch_value.detach()
        losses["L_pitch"] = pitch_value
        losses["L_distill_exec"] = losses.get("rhythm_distill_exec", zero).detach() if isinstance(losses.get("rhythm_distill_exec"), torch.Tensor) else zero
        losses["L_distill_budget"] = losses.get("rhythm_distill_budget", zero).detach() if isinstance(losses.get("rhythm_distill_budget"), torch.Tensor) else zero
        losses["L_distill_prefix"] = losses.get("rhythm_distill_prefix", zero).detach() if isinstance(losses.get("rhythm_distill_prefix"), torch.Tensor) else zero
        base_value = losses.get("base")
        if isinstance(base_value, torch.Tensor):
            base = base_value.detach()
        else:
            base = zero
            for loss_name in self.mel_losses.keys():
                value = losses.get(loss_name)
                if isinstance(value, torch.Tensor):
                    base = base + value.detach()
        losses["L_base"] = base

    @staticmethod
    def _detach_loss_value(losses, key):
        value = losses.get(key)
        if isinstance(value, torch.Tensor):
            losses[key] = value.detach()
        return value

    def _compact_base_optimizer_losses(self, losses, *, schedule_only_stage: bool):
        if schedule_only_stage or not bool(hparams.get("rhythm_compact_joint_loss", True)):
            return
        base_terms = []
        base_keys = []
        for loss_name in self.mel_losses.keys():
            value = losses.get(loss_name)
            if isinstance(value, torch.Tensor):
                base_keys.append(loss_name)
                if value.requires_grad:
                    base_terms.append(value)
        if not base_terms:
            return
        losses["base"] = sum(base_terms)
        for key in base_keys:
            self._detach_loss_value(losses, key)

    def _compact_pitch_optimizer_losses(self, losses, *, schedule_only_stage: bool):
        if schedule_only_stage or not bool(hparams.get("rhythm_compact_joint_loss", True)):
            return
        pitch_keys = []
        pitch_terms = []
        for key in ("fdiff", "uv", "pflow", "gdiff", "mdiff"):
            value = losses.get(key)
            if isinstance(value, torch.Tensor):
                pitch_keys.append(key)
                if value.requires_grad:
                    pitch_terms.append(value)
        if not pitch_terms:
            return
        losses["pitch"] = sum(pitch_terms)
        for key in pitch_keys:
            self._detach_loss_value(losses, key)

    def _compact_rhythm_optimizer_losses(self, losses, *, schedule_only_stage: bool):
        # Keep these available for logging/ablation, but keep them out of optimizer by default.
        if not bool(hparams.get("rhythm_enable_aux_optimizer_losses", False)):
            for key in ("rhythm_plan", "rhythm_guidance", "rhythm_distill"):
                self._detach_loss_value(losses, key)
        if schedule_only_stage or not bool(hparams.get("rhythm_compact_joint_loss", True)):
            return
        exec_terms = []
        exec_keys = []
        for key in ("rhythm_exec_speech", "rhythm_exec_pause"):
            value = losses.get(key)
            if isinstance(value, torch.Tensor):
                exec_keys.append(key)
                if value.requires_grad:
                    exec_terms.append(value)
        if exec_terms:
            losses["rhythm_exec"] = sum(exec_terms)
            for key in exec_keys:
                self._detach_loss_value(losses, key)
        budget = losses.get("rhythm_budget")
        cumplan = losses.get("rhythm_cumplan", losses.get("rhythm_carry"))
        state_terms = []
        if isinstance(budget, torch.Tensor) and budget.requires_grad:
            state_terms.append(float(hparams.get("rhythm_joint_budget_macro_weight", 0.35)) * budget)
        if isinstance(cumplan, torch.Tensor) and cumplan.requires_grad:
            state_terms.append(float(hparams.get("rhythm_joint_cumplan_macro_weight", 0.65)) * cumplan)
        if state_terms:
            losses["rhythm_stream_state"] = sum(state_terms)
        self._detach_loss_value(losses, "rhythm_budget")
        if "rhythm_cumplan" in losses:
            self._detach_loss_value(losses, "rhythm_cumplan")
        elif "rhythm_carry" in losses:
            self._detach_loss_value(losses, "rhythm_carry")

    def _add_acoustic_loss(self, mel_out, target, losses, *, frame_weight=None):
        if frame_weight is None:
            self.add_mel_loss(mel_out, target, losses)
            return
        for loss_name, lambd in self.mel_losses.items():
            if loss_name == "l1":
                loss = self._weighted_l1_loss(mel_out, target, frame_weight)
            elif loss_name == "mse":
                loss = self._weighted_mse_loss(mel_out, target, frame_weight)
            elif loss_name == "ssim":
                loss = self._weighted_ssim_loss(mel_out, target, frame_weight)
            else:
                loss = getattr(self, f'{loss_name}_loss')(mel_out, target)
            losses[loss_name] = loss * lambd

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech
            
    def run_model(self, sample, infer=False, test=False, **kwargs):
        # txt_tokens = sample["txt_tokens"]
        # mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        content = sample["content"]
        if 'spk_embed' in sample:
            spk_embed = sample["spk_embed"]
        else:
            spk_embed=None
        f0, uv = sample.get("f0", None), sample.get("uv", None)
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]

        target = sample["mels"]
        effective_global_step = 200000 if test else int(self.global_step)
        use_reference = (
            test
            or effective_global_step >= hparams["random_speaker_steps"]
            or bool(hparams.get("rhythm_force_reference_conditioning", False))
        )
        if use_reference:
            ref = sample['ref_mels']
        else:
            ref = target
        rhythm_apply_override = self._resolve_rhythm_apply_override(
            infer=infer,
            test=test,
            explicit=kwargs.get("rhythm_apply_override"),
            current_step=effective_global_step,
        )
        apply_rhythm_render = resolve_rhythm_apply_mode(
            hparams,
            infer=infer,
            override=rhythm_apply_override,
        )
        retimed_stage_active = bool(
            apply_rhythm_render
            and not infer
            and not test
            and bool(hparams.get("rhythm_use_retimed_target_if_available", False))
            and effective_global_step >= int(hparams.get("rhythm_retimed_target_start_steps", 0) or 0)
        )
        use_retimed_pitch_target = bool(hparams.get("rhythm_use_retimed_pitch_target", False))
        disable_source_pitch_supervision = bool(
            retimed_stage_active
            and (
                not use_retimed_pitch_target
                or f0 is None
                or uv is None
                or bool(hparams.get("rhythm_disable_pitch_loss_when_retimed", False))
            )
        )
        if disable_source_pitch_supervision and retimed_stage_active and not self._warned_retimed_pitch_supervision:
            print("| Rhythm V2: retimed canvas active, disabling source-aligned pitch supervision for this run.")
            self._warned_retimed_pitch_supervision = True
        rhythm_ref_conditioning = kwargs.get("rhythm_ref_conditioning")
        if rhythm_ref_conditioning is None:
            ref_stats = sample.get("ref_rhythm_stats")
            ref_trace = sample.get("ref_rhythm_trace")
            if ref_stats is not None and ref_trace is not None:
                rhythm_ref_conditioning = {
                    "ref_rhythm_stats": ref_stats,
                    "ref_rhythm_trace": ref_trace,
                }
                for extra_key in (
                    "slow_rhythm_memory",
                    "slow_rhythm_summary",
                    "selector_meta_indices",
                    "selector_meta_scores",
                    "selector_meta_starts",
                    "selector_meta_ends",
                ):
                    extra_value = sample.get(extra_key)
                    if extra_value is not None:
                        rhythm_ref_conditioning[extra_key] = extra_value
        # assert False, f'content: {content.shape}, target: {target.shape},spk_embed: {spk_embed.shape}'
        # if not infer:
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathy': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1,
        #         'glissando': 0.1,
        #         'pharyngeal': 0.1,
        #         'vibrato': 0.1,
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # mix, falsetto, breathy=sample['mix'], sample['falsetto'], sample['breathy']
        # bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
        # pharyngeal, vibrato, glissando = sample['pharyngeal'], sample['vibrato'], sample['glissando']
        output = self.model(content,spk_embed=spk_embed, target=target,ref=ref,
                            f0=None if disable_source_pitch_supervision else f0,
                            uv=None if disable_source_pitch_supervision else uv,
                            infer=infer, global_steps=effective_global_step,
                            content_lengths=sample.get("mel_lengths"),
                            ref_lengths=sample.get("ref_mel_lengths"),
                            rhythm_apply_override=rhythm_apply_override,
                            rhythm_state=kwargs.get("rhythm_state"),
                            rhythm_ref_conditioning=rhythm_ref_conditioning,
                            rhythm_source_cache=self._collect_rhythm_source_cache(sample),
                            rhythm_offline_source_cache=self._collect_rhythm_source_cache(sample, prefix="rhythm_offline_"))
        acoustic_target, acoustic_target_is_retimed, acoustic_weight, acoustic_target_source = self._resolve_acoustic_target_post_model(
            sample,
            output,
            apply_rhythm_render=bool(apply_rhythm_render),
            infer=infer,
            test=test,
            current_step=effective_global_step,
        )
        rhythm_execution = output.get("rhythm_execution")
        if rhythm_execution is not None and getattr(rhythm_execution, "planner", None) is not None:
            planner = rhythm_execution.planner
            for attr_name in (
                "feasible_speech_budget_delta",
                "feasible_pause_budget_delta",
                "feasible_total_budget_delta",
            ):
                attr_value = getattr(planner, attr_name, None)
                if attr_value is not None:
                    output[attr_name] = attr_value
        
        losses = {}
        output["acoustic_target_mel"] = acoustic_target
        output["acoustic_target_is_retimed"] = bool(acoustic_target_is_retimed)
        output["acoustic_target_weight"] = acoustic_weight
        output["acoustic_target_source"] = acoustic_target_source
        output["rhythm_pitch_supervision_disabled"] = float(disable_source_pitch_supervision)
        if acoustic_target_is_retimed:
            mel_out_aligned, acoustic_target, acoustic_weight = self._align_acoustic_target_to_output(
                output['mel_out'],
                acoustic_target,
                acoustic_weight,
            )
            output['mel_out'] = mel_out_aligned
            output["acoustic_target_mel"] = acoustic_target
            output["acoustic_target_weight"] = acoustic_weight
        
        if not test:
            schedule_only_stage = bool(hparams.get("rhythm_schedule_only_stage", False))
            output["rhythm_schedule_only_stage"] = float(schedule_only_stage)
            if not schedule_only_stage:
                self._add_acoustic_loss(output['mel_out'], acoustic_target, losses, frame_weight=acoustic_weight)
                # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
                self.add_pitch_loss(output, sample, losses)
            self.add_rhythm_loss(output, sample, losses)
            if (
                hparams['style']
                and not schedule_only_stage
                and not getattr(self.model, "rhythm_minimal_style_only", False)
            ):
                if (
                    self.global_step > hparams['forcing']
                    and self.global_step < hparams['random_speaker_steps']
                    and 'gloss' in output
                ):
                    losses['gloss'] = output['gloss']
                if self.global_step > hparams['vq_start'] and 'vq_loss' in output and 'ppl' in output:
                    losses['vq_loss'] = output['vq_loss']
                    losses['ppl'] = output['ppl']
            self._compact_base_optimizer_losses(losses, schedule_only_stage=schedule_only_stage)
            self._compact_pitch_optimizer_losses(losses, schedule_only_stage=schedule_only_stage)
            self._compact_rhythm_optimizer_losses(losses, schedule_only_stage=schedule_only_stage)
            self._update_public_loss_aliases(losses)
        
        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        if bool(output.get("rhythm_pitch_supervision_disabled", False)):
            return
        content = output.get("content", sample['content'])
        f0 = output.get("retimed_f0_tgt", sample.get('f0'))
        uv = output.get("retimed_uv_tgt", sample.get('uv'))
        if f0 is None or uv is None:
            return
        nonpadding = (content != -1).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
        elif hparams["f0_gen"] == "orig":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']

    def _build_rhythm_loss_targets(self, output, sample):
        if "rhythm_execution" not in output or output["rhythm_execution"] is None:
            return None
        runtime_teacher = output.get("rhythm_offline_execution")
        algorithmic_teacher = output.get("rhythm_algorithmic_teacher")
        lambda_guidance = float(hparams.get("lambda_rhythm_guidance", 0.0))
        lambda_distill = float(hparams.get("lambda_rhythm_distill", 0.0))
        distill_budget_weight = float(hparams.get("rhythm_distill_budget_weight", 0.5))
        distill_allocation_weight = float(hparams.get("rhythm_distill_allocation_weight", 0.5))
        distill_prefix_weight = float(hparams.get("rhythm_distill_prefix_weight", 0.25))
        use_guidance = lambda_guidance > 0.0
        use_distill = lambda_distill > 0.0
        use_distill_budget = use_distill and distill_budget_weight > 0.0
        use_distill_allocation = use_distill and distill_allocation_weight > 0.0
        use_distill_prefix = use_distill and distill_prefix_weight > 0.0
        primary_surface = self._resolve_rhythm_primary_target_surface()
        distill_surface = self._resolve_rhythm_distill_surface()
        pause_exec_key = "rhythm_pause_exec_tgt" if "rhythm_pause_exec_tgt" in sample else "rhythm_blank_exec_tgt"
        pause_budget_key = "rhythm_pause_budget_tgt" if "rhythm_pause_budget_tgt" in sample else "rhythm_blank_budget_tgt"
        guidance_pause_key = "rhythm_guidance_pause_tgt" if "rhythm_guidance_pause_tgt" in sample else "rhythm_guidance_blank_tgt"
        distill_pause_key = (
            "rhythm_teacher_pause_exec_tgt"
            if "rhythm_teacher_pause_exec_tgt" in sample
            else "rhythm_teacher_blank_exec_tgt"
        )
        distill_pause_budget_key = (
            "rhythm_teacher_pause_budget_tgt"
            if "rhythm_teacher_pause_budget_tgt" in sample
            else "rhythm_teacher_blank_budget_tgt"
        )
        if primary_surface == "teacher":
            target_speech_key = "rhythm_teacher_speech_exec_tgt"
            target_pause_key = distill_pause_key
            target_speech_budget_key = "rhythm_teacher_speech_budget_tgt"
            target_pause_budget_key = distill_pause_budget_key
            sample_confidence = sample.get("rhythm_teacher_confidence", sample.get("rhythm_target_confidence"))
        else:
            target_speech_key = "rhythm_speech_exec_tgt"
            target_pause_key = pause_exec_key
            target_speech_budget_key = "rhythm_speech_budget_tgt"
            target_pause_budget_key = pause_budget_key
            sample_confidence = sample.get("rhythm_target_confidence")
        explicit_keys = [
            target_speech_key,
            target_pause_key,
            target_speech_budget_key,
            target_pause_budget_key,
        ]
        if all(key in sample for key in explicit_keys):
            unit_batch = output["rhythm_unit_batch"]
            guidance_speech = sample.get("rhythm_guidance_speech_tgt") if use_guidance else None
            guidance_pause = sample.get(guidance_pause_key) if use_guidance else None
            distill_speech = None
            distill_pause = None
            distill_speech_budget = None
            distill_pause_budget = None
            distill_allocation = None
            distill_prefix_clock = None
            distill_prefix_backlog = None
            distill_confidence = None
            distill_exec_confidence = None
            distill_budget_confidence = None
            distill_prefix_confidence = None
            distill_allocation_confidence = None

            if use_distill and distill_surface in {"auto", "cache"}:
                distill_speech = sample.get("rhythm_teacher_speech_exec_tgt")
                distill_pause = sample.get(distill_pause_key)
                distill_speech_budget = sample.get("rhythm_teacher_speech_budget_tgt") if use_distill_budget else None
                distill_pause_budget = sample.get(distill_pause_budget_key) if use_distill_budget else None
                distill_allocation = sample.get("rhythm_teacher_allocation_tgt") if use_distill_allocation else None
                distill_prefix_clock = sample.get("rhythm_teacher_prefix_clock_tgt") if use_distill_prefix else None
                distill_prefix_backlog = sample.get("rhythm_teacher_prefix_backlog_tgt") if use_distill_prefix else None
                distill_confidence = sample.get("rhythm_teacher_confidence")
            if use_distill and distill_speech is None and distill_surface in {"auto", "offline"} and runtime_teacher is not None:
                distill_speech = runtime_teacher.speech_duration_exec.detach()
                distill_pause = getattr(runtime_teacher, "blank_duration_exec", runtime_teacher.pause_after_exec).detach()
                (
                    distill_speech,
                    distill_pause,
                    distill_speech_budget,
                    distill_pause_budget,
                    distill_allocation,
                    distill_prefix_clock,
                    distill_prefix_backlog,
                ) = self._slice_rhythm_surface_to_student(
                    speech_exec=distill_speech,
                    pause_exec=distill_pause,
                    student_units=unit_batch.dur_anchor_src.size(1),
                    dur_anchor_src=unit_batch.dur_anchor_src,
                    unit_mask=unit_batch.unit_mask,
                )
                if not use_distill_budget:
                    distill_speech_budget = None
                    distill_pause_budget = None
                if not use_distill_allocation:
                    distill_allocation = None
                if not use_distill_prefix:
                    distill_prefix_clock = None
                    distill_prefix_backlog = None
                distill_confidence = output.get("rhythm_offline_confidence")
                distill_exec_confidence = output.get("rhythm_offline_confidence_exec")
                distill_budget_confidence = output.get("rhythm_offline_confidence_budget")
                distill_prefix_confidence = output.get("rhythm_offline_confidence_prefix")
                distill_allocation_confidence = output.get("rhythm_offline_confidence_allocation")
                if distill_confidence is None:
                    distill_confidence = distill_speech.new_ones((distill_speech.size(0), 1))
            if use_distill and distill_speech is None and distill_surface in {"auto", "algorithmic"} and algorithmic_teacher is not None:
                distill_speech = algorithmic_teacher.speech_exec_tgt.detach()
                distill_pause = algorithmic_teacher.pause_exec_tgt.detach()
                distill_speech_budget = algorithmic_teacher.speech_budget_tgt.detach() if use_distill_budget else None
                distill_pause_budget = algorithmic_teacher.pause_budget_tgt.detach() if use_distill_budget else None
                distill_allocation = algorithmic_teacher.allocation_tgt.detach() if use_distill_allocation else None
                distill_prefix_clock = algorithmic_teacher.prefix_clock_tgt.detach() if use_distill_prefix else None
                distill_prefix_backlog = algorithmic_teacher.prefix_backlog_tgt.detach() if use_distill_prefix else None
                distill_confidence = algorithmic_teacher.confidence.detach()
            elif distill_speech is not None and distill_pause is not None:
                if distill_speech.size(1) != unit_batch.dur_anchor_src.size(1):
                    (
                        distill_speech,
                        distill_pause,
                        distill_speech_budget,
                        distill_pause_budget,
                        distill_allocation,
                        distill_prefix_clock,
                        distill_prefix_backlog,
                    ) = self._slice_rhythm_surface_to_student(
                        speech_exec=distill_speech,
                        pause_exec=distill_pause,
                        student_units=unit_batch.dur_anchor_src.size(1),
                        dur_anchor_src=unit_batch.dur_anchor_src,
                        unit_mask=unit_batch.unit_mask,
                    )
                if use_distill_allocation and distill_allocation is None:
                    distill_allocation = (distill_speech.float() + distill_pause.float())
                if use_distill_prefix and (distill_prefix_clock is None or distill_prefix_backlog is None):
                    distill_prefix_clock, distill_prefix_backlog = self._build_prefix_carry_from_exec(
                        distill_speech,
                        distill_pause,
                        unit_batch.dur_anchor_src,
                        unit_batch.unit_mask,
                    )
            if use_distill:
                distill_confidence = self._normalize_distill_confidence(
                    distill_confidence,
                    batch_size=unit_batch.dur_anchor_src.size(0),
                    device=unit_batch.dur_anchor_src.device,
                )
                distill_exec_confidence = self._normalize_component_distill_confidence(
                    distill_exec_confidence,
                    fallback_confidence=distill_confidence,
                    batch_size=unit_batch.dur_anchor_src.size(0),
                    device=unit_batch.dur_anchor_src.device,
                )
                distill_budget_confidence = self._normalize_component_distill_confidence(
                    distill_budget_confidence,
                    fallback_confidence=distill_confidence,
                    batch_size=unit_batch.dur_anchor_src.size(0),
                    device=unit_batch.dur_anchor_src.device,
                )
                distill_prefix_confidence = self._normalize_component_distill_confidence(
                    distill_prefix_confidence,
                    fallback_confidence=distill_confidence,
                    batch_size=unit_batch.dur_anchor_src.size(0),
                    device=unit_batch.dur_anchor_src.device,
                )
                distill_allocation_confidence = self._normalize_component_distill_confidence(
                    distill_allocation_confidence,
                    fallback_confidence=distill_confidence,
                    batch_size=unit_batch.dur_anchor_src.size(0),
                    device=unit_batch.dur_anchor_src.device,
                )
            return RhythmLossTargets(
                speech_exec_tgt=sample[target_speech_key],
                pause_exec_tgt=sample[target_pause_key],
                speech_budget_tgt=sample[target_speech_budget_key],
                pause_budget_tgt=sample[target_pause_budget_key],
                unit_mask=unit_batch.unit_mask,
                dur_anchor_src=unit_batch.dur_anchor_src,
                plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
                plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
                sample_confidence=sample_confidence,
                guidance_speech_tgt=guidance_speech,
                guidance_pause_tgt=guidance_pause,
                guidance_confidence=sample.get("rhythm_guidance_confidence"),
                distill_speech_tgt=distill_speech,
                distill_pause_tgt=distill_pause,
                distill_speech_budget_tgt=distill_speech_budget,
                distill_pause_budget_tgt=distill_pause_budget,
                distill_allocation_tgt=distill_allocation,
                distill_prefix_clock_tgt=distill_prefix_clock,
                distill_prefix_backlog_tgt=distill_prefix_backlog,
                distill_confidence=distill_confidence,
                distill_exec_confidence=distill_exec_confidence,
                distill_budget_confidence=distill_budget_confidence,
                distill_prefix_confidence=distill_prefix_confidence,
                distill_allocation_confidence=distill_allocation_confidence,
                distill_budget_weight=distill_budget_weight,
                distill_allocation_weight=distill_allocation_weight,
                distill_prefix_weight=distill_prefix_weight,
                pause_boundary_weight=self._resolve_rhythm_pause_boundary_weight(),
                feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
            )
        if not hparams.get("rhythm_train_identity_fallback", False):
            return None
        unit_batch = output.get("rhythm_unit_batch")
        if unit_batch is None:
            return None
        unit_mask = unit_batch.unit_mask.float()
        speech_exec_tgt = unit_batch.dur_anchor_src.float() * unit_mask
        pause_exec_tgt = torch.zeros_like(speech_exec_tgt)
        speech_budget_tgt = speech_exec_tgt.sum(dim=1, keepdim=True)
        pause_budget_tgt = torch.zeros_like(speech_budget_tgt)
        return RhythmLossTargets(
            speech_exec_tgt=speech_exec_tgt,
            pause_exec_tgt=pause_exec_tgt,
            speech_budget_tgt=speech_budget_tgt,
            pause_budget_tgt=pause_budget_tgt,
            unit_mask=unit_mask,
            dur_anchor_src=unit_batch.dur_anchor_src,
            plan_local_weight=float(hparams.get("rhythm_plan_local_weight", 0.5)),
            plan_cum_weight=float(hparams.get("rhythm_plan_cum_weight", 1.0)),
            sample_confidence=torch.ones((unit_mask.size(0), 1), device=unit_mask.device),
            distill_budget_weight=distill_budget_weight,
            distill_allocation_weight=distill_allocation_weight,
            distill_prefix_weight=distill_prefix_weight,
            pause_boundary_weight=self._resolve_rhythm_pause_boundary_weight(),
            feasible_debt_weight=float(hparams.get("rhythm_feasible_debt_weight", 0.05)),
        )

    def add_rhythm_loss(self, output, sample, losses):
        targets = self._build_rhythm_loss_targets(output, sample)
        if targets is None:
            return
        rhythm_execution = output["rhythm_execution"]
        if rhythm_execution is not None and "rhythm_pause_exec_surrogate_used" not in output:
            output["rhythm_pause_exec_surrogate_used"] = 0.0
        rhythm_losses = build_rhythm_loss_dict(rhythm_execution, targets)
        losses["rhythm_exec_speech"] = rhythm_losses["rhythm_exec_speech"] * hparams.get("lambda_rhythm_exec_speech", 1.0)
        losses["rhythm_exec_pause"] = rhythm_losses["rhythm_exec_pause"] * hparams.get("lambda_rhythm_exec_pause", 1.0)
        losses["rhythm_budget"] = rhythm_losses["rhythm_budget"] * hparams.get("lambda_rhythm_budget", 0.25)
        losses["rhythm_feasible_debt"] = (
            rhythm_losses["rhythm_feasible_debt"]
            * hparams.get("lambda_rhythm_budget", 0.25)
            * float(hparams.get("rhythm_feasible_debt_weight", 0.05))
        ).detach()
        losses["rhythm_cumplan"] = rhythm_losses["rhythm_cumplan"] * self._get_rhythm_cumplan_lambda()
        losses["rhythm_plan"] = rhythm_losses["rhythm_plan"] * hparams.get("lambda_rhythm_plan", 0.0)
        losses["rhythm_guidance"] = rhythm_losses["rhythm_guidance"] * hparams.get("lambda_rhythm_guidance", 0.0)
        losses["rhythm_distill"] = rhythm_losses["rhythm_distill"] * hparams.get("lambda_rhythm_distill", 0.0)
        lambda_distill = hparams.get("lambda_rhythm_distill", 0.0)
        losses["rhythm_distill_exec"] = (rhythm_losses["rhythm_distill_exec"] * lambda_distill).detach()
        losses["rhythm_distill_budget"] = (
            rhythm_losses["rhythm_distill_budget"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_budget_weight", 0.5))
        ).detach()
        losses["rhythm_distill_prefix"] = (
            rhythm_losses["rhythm_distill_prefix"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_prefix_weight", 0.25))
        ).detach()
        losses["rhythm_distill_allocation"] = (
            rhythm_losses["rhythm_distill_allocation"]
            * lambda_distill
            * float(hparams.get("rhythm_distill_allocation_weight", 0.5))
        ).detach()

            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            adv_disabled = bool(
                model_out.get("acoustic_target_is_retimed", False)
                and hparams.get("rhythm_disable_mel_adv_when_retimed", True)
            )
            self._disc_skip_for_retimed = adv_disabled
            if disc_start and not adv_disabled:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                adv_disabled = bool(getattr(self, "_disc_skip_for_retimed", False))
                if not adv_disabled:
                    mel_g = model_out.get('acoustic_target_mel', sample['mels'])
                    mel_p = model_out['mel_out']
                    o = self.mel_disc(mel_g)
                    p, pc = o['y'], o['y_c']
                    o_ = self.mel_disc(mel_p)
                    p_, pc_ = o_['y'], o_['y_c']
                    if p_ is not None:
                        loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                        loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                    if pc_ is not None:
                        loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                        loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(loss_output) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['content'].size()[0]
        return total_loss, loss_output


    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        outputs['rhythm_metrics'] = tensors_to_scalars(build_rhythm_metric_dict(model_out, sample))
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            acoustic_target = model_out.get("acoustic_target_mel", sample["mels"])
            acoustic_target_is_retimed = bool(model_out.get("acoustic_target_is_retimed", False))
            if acoustic_target_is_retimed:
                wav_gt = self.vocoder.spec2wav(acoustic_target[0].cpu().numpy())
            else:
                wav_gt = self.vocoder.spec2wav(acoustic_target[0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=model_out["f0_denorm_pred"][0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, acoustic_target, model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(None if acoustic_target_is_retimed else gt_f0[0], None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs
    
    def test_step(self, sample, batch_idx):
        if "ref_mels" not in sample or sample["ref_mels"] is None:
            sample["ref_mels"] = sample["mels"]
        if hparams.get("rhythm_test_chunkwise", True):
            stream_result = run_chunkwise_streaming_inference(
                self,
                sample,
                tokens_per_chunk=int(hparams.get("rhythm_test_tokens_per_chunk", 4)),
            )
            outputs = stream_result.final_output
            mel_pred = stream_result.mel_pred
            stream_metrics = build_streaming_chunk_metrics(stream_result)
        else:
            outputs = self.run_model(sample, infer=True, test=True)[1]
            mel_pred = outputs['mel_out'][0]
            stream_metrics = {}
        item_name = sample['item_name'][0]
        base_fn = f'{item_name.replace(" ", "_")}[P]'

        # Pass through vocoder at once
        wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())

        # Optional: save gt (keep consistent with original implementation)
        if hparams.get('save_gt', False):
            mel_gt = sample['mels'][0].cpu().numpy()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_gt, mel_gt,
                    base_fn.replace('[P]', '[G]'),
                    self.gen_dir, None, None,
                    denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                    None, None]
            )

        # Save prediction
        self.saving_result_pool.add_job(
            self.save_result,
            args=[wav_pred, mel_pred.cpu().numpy(),
                base_fn, self.gen_dir, None, None,
                denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                outputs.get('f0_denorm_pred')[0].cpu().numpy()
                if outputs.get('f0_denorm_pred') is not None else None,
                None]
        )

        result = {}
        result.update(tensors_to_scalars(build_rhythm_metric_dict(outputs, sample)))
        result.update(stream_metrics)
        return result


    def build_optimizer(self, model):
        
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            super().build_scheduler( optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, (list, tuple)):
            if 0 <= optimizer_idx < len(self.scheduler) and self.scheduler[optimizer_idx] is not None:
                self.scheduler[optimizer_idx].step(self.global_step // hparams['accumulate_grad_batches'])
            return
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result

class VCPostnetTask(ConanTask):
    def __init__(self):
        super(VCPostnetTask, self).__init__()
        self.drop_prob=hparams['drop_tech_prob']

    def build_model(self):
        self.build_pretrain_model()
        self.model = ConanPostnet()

    def build_pretrain_model(self):
        dict_size = 0
        self.pretrain = Conan(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    
    
    def run_model(self, sample, infer=False, noise=None,test=False):
        content = sample["content"]
        # spk_embed = sample["spk_embed"]
        spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
        target = sample["mels"]
        ref=sample['ref_mels']
        cfg = False
        output = self.pretrain(content,spk_embed=spk_embed, target=target,ref=ref,
                f0=f0, uv=uv,
                infer=infer)

        self.model(target, infer, output, cfg, cfg_scale=hparams['cfg_scale'],  noise=noise)
        losses = {}
        losses["flow"] = output["flow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

# class TechSingerTask(RFSingerTask):

#     def build_tts_model(self):
#         dict_size = len(self.token_encoder)
#         self.model = Conan(dict_size, hparams)
#         self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
            
#     def run_model(self, sample, infer=False):
#         txt_tokens = sample["txt_tokens"]
#         mel2ph = sample["mel2ph"]
#         spk_id = sample["spk_ids"]
#         f0, uv = sample["f0"], sample["uv"]
#         notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
#         target = sample["mels"]
#         cfg = False
        
#         if infer:
#             cfg = True
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
#             ububble, ustrong, uweak = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(strong, dtype=strong.dtype) * 2, torch.ones_like(weak, dtype=weak.dtype) * 2
#             upharyngeal, uvibrato, uglissando = torch.ones_like(bubble, dtype=bubble.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
#             mix = torch.cat((mix, umix), dim=0)
#             falsetto = torch.cat((falsetto, ufalsetto), dim=0)
#             breathy = torch.cat((breathy, ubreathy), dim=0)
#             bubble = torch.cat((bubble, ububble), dim=0)
#             strong = torch.cat((strong, ustrong), dim=0)
#             weak = torch.cat((weak, uweak), dim=0)
#             pharyngeal = torch.cat((pharyngeal, upharyngeal), dim=0)
#             vibrato = torch.cat((vibrato, uvibrato), dim=0)
#             glissando = torch.cat((glissando, uglissando), dim=0)
            
#             txt_tokens = self_clone(txt_tokens)
#             mel2ph = self_clone(mel2ph)
#             spk_id = self_clone(spk_id)
#             f0 = self_clone(f0)
#             uv = self_clone(uv)
#             notes = self_clone(notes)
#             note_durs = self_clone(note_durs)
#             note_types = self_clone(note_types)
            
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg, cfg_scale=1.0,
#                                 infer=infer)
#         else:
#             tech_drop = {
#                 'mix': 0.1,
#                 'falsetto': 0.1,
#                 'breathy': 0.1,
#                 'bubble': 0.1,
#                 'strong': 0.1,
#                 'weak': 0.1,
#                 'glissando': 0.1,
#                 'pharyngeal': 0.1,
#                 'vibrato': 0.1,
#             }
#             for tech, drop_p in tech_drop.items():
#                 sample[tech] = self.drop_multi(sample[tech], drop_p)
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg,
#                                 infer=infer)
        
#         losses = {}
#         if not infer:
#             self.add_mel_loss(output['coarse_mel_out'], target, losses)
#             self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
#             self.add_pitch_loss(output, sample, losses)
#         losses["flow"] = output["flow"]
#         return losses, output
