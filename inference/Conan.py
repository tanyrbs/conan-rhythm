import os
from typing import Any, Dict, List

import numpy as np
import torch
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.Conan.Conan import Conan
from modules.Conan.rhythm_v3.contracts import export_duration_v3_source_cache
from modules.Emformer.emformer import EmformerDistillModel
from tasks.Conan.rhythm.streaming_commit import extract_incremental_committed_mel
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams

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
        self.runtime_metadata = self.get_streaming_runtime_metadata()
        self.last_inference_metadata = dict(self.runtime_metadata)
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

    def _extract_prompt_unit_conditioning(
        self,
        ref_mel_batch: torch.Tensor,
        prepared_spk_embed: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            ref_lengths = torch.tensor([ref_mel_batch.size(1)], dtype=torch.long, device=ref_mel_batch.device)
            codes, _, _ = self.emformer.emformer.infer(ref_mel_batch, ref_lengths, None)
            if self.emformer.mode == "both":
                codes = self.emformer.proj1(codes)
            else:
                codes = self.emformer.proj(codes)
            if codes.dim() == 3 and codes.shape[-1] > 1:
                codes = torch.argmax(codes, dim=-1)

        from modules.Conan.rhythm.supervision import build_source_rhythm_cache

        cache = build_source_rhythm_cache(
            codes[0].detach().cpu().tolist(),
            silent_token=self.hparams.get("silent_token", 57),
            separator_aware=bool(self.hparams.get("rhythm_separator_aware", True)),
            tail_open_units=int(self.hparams.get("rhythm_tail_open_units", 1)),
            emit_silence_runs=bool(self.hparams.get("rhythm_v3_emit_silence_runs", True)),
            phrase_boundary_threshold=float(self.hparams.get("rhythm_source_phrase_threshold", 0.55)),
        )
        prompt_content_units = torch.tensor(cache["content_units"], device=self.device, dtype=torch.long).unsqueeze(0)
        prompt_duration_obs = torch.tensor(cache["dur_anchor_src"], device=self.device, dtype=torch.float32).unsqueeze(0)
        prompt_valid_mask = (prompt_duration_obs > 0).float()
        prompt_silence = cache.get("source_silence_mask")
        if prompt_silence is not None:
            prompt_silence_mask = torch.tensor(prompt_silence, device=self.device, dtype=torch.float32).unsqueeze(0)
            prompt_speech_mask = prompt_valid_mask * (1.0 - prompt_silence_mask.clamp(0.0, 1.0))
        else:
            prompt_speech_mask = prompt_valid_mask
        out = {
            "prompt_content_units": prompt_content_units,
            "prompt_duration_obs": prompt_duration_obs,
            "prompt_unit_mask": prompt_valid_mask,
            "prompt_valid_mask": prompt_valid_mask,
            "prompt_speech_mask": prompt_speech_mask,
        }
        if prepared_spk_embed is not None:
            prompt_spk = prepared_spk_embed
            if prompt_spk.dim() == 3 and prompt_spk.size(-1) == 1:
                prompt_spk = prompt_spk.squeeze(-1)
            elif prompt_spk.dim() == 3 and prompt_spk.size(1) == 1:
                prompt_spk = prompt_spk.squeeze(1)
            if prompt_spk.dim() == 2:
                out["prompt_spk_embed"] = prompt_spk.detach()
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

    def infer_once(self, inp: Dict, spk_embed=None, return_metadata: bool = False):
        if hasattr(self.vocoder, "reset_stream"):
            self.vocoder.reset_stream()

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

            with torch.no_grad():
                out = self.model(
                    content=all_codes,
                    spk_embed=prepared_spk_embed,
                    target=None,
                    ref=ref_mel_batch if require_runtime_ref else None,
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=200000,
                    content_lengths=torch.tensor([all_codes.size(1)], device=self.device),
                    rhythm_state=rhythm_state,
                    rhythm_ref_conditioning=rhythm_ref_conditioning,
                    rhythm_source_cache=rhythm_source_cache,
                    decoder_cache=decoder_cache,
                )
                rhythm_state = out.get("rhythm_state_next", rhythm_state)
                rhythm_ref_conditioning = out.get("rhythm_ref_conditioning", rhythm_ref_conditioning)
                decoder_cache = out.get("decoder_cache", decoder_cache)
                last_mel_out = out["mel_out"][0]

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

        if prev_committed_len < int(last_mel_out.shape[0]):
            mel_tail = last_mel_out[prev_committed_len:]
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
                "full_end_to_end_stateful_streaming": False,
            }
        )

        if return_metadata:
            return wav_pred, mel_pred.cpu().numpy(), self.last_inference_metadata
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
