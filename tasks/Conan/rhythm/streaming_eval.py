from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StreamingEvalResult:
    mel_pred: torch.Tensor
    final_output: dict
    mel_lengths: list[int]
    commit_history: list[list[int]]


def run_chunkwise_streaming_inference(task, sample, *, tokens_per_chunk: int = 4) -> StreamingEvalResult:
    sample = {k: v for k, v in sample.items()}
    if "ref_mels" not in sample or sample["ref_mels"] is None:
        sample["ref_mels"] = sample["mels"]

    content_full = sample["content"]
    if content_full.dim() != 2:
        raise ValueError(f"Expected batched content [B,T], got {tuple(content_full.shape)}")
    batch_size, total_tokens = content_full.shape
    if batch_size != 1:
        raise ValueError("Chunkwise test_step currently expects batch_size == 1.")

    total_chunks = max(1, (total_tokens + tokens_per_chunk - 1) // tokens_per_chunk)
    mel_chunks = []
    mel_lengths = []
    commit_history = []
    prev_mel_len = 0
    rhythm_state = None
    rhythm_ref_conditioning = None
    final_output = None

    for chunk_idx in range(total_chunks):
        end_idx = min((chunk_idx + 1) * tokens_per_chunk, total_tokens)
        sample_chunk = {k: v for k, v in sample.items()}
        sample_chunk["content"] = content_full[:, :end_idx]
        sample_chunk["mel_lengths"] = torch.tensor([end_idx], dtype=torch.long, device=content_full.device)
        losses, outputs = task.run_model(
            sample_chunk,
            infer=True,
            test=True,
            rhythm_state=rhythm_state,
            rhythm_ref_conditioning=rhythm_ref_conditioning,
        )
        del losses
        rhythm_state = outputs.get("rhythm_state_next", rhythm_state)
        rhythm_ref_conditioning = outputs.get("rhythm_ref_conditioning", rhythm_ref_conditioning)
        mel_out = outputs["mel_out"][0]
        mel_new = mel_out[prev_mel_len:]
        if mel_new.numel() > 0:
            mel_chunks.append(mel_new)
        prev_mel_len = int(mel_out.shape[0])
        mel_lengths.append(prev_mel_len)
        commit = outputs.get("commit_frontier")
        commit_history.append(commit.detach().cpu().tolist() if commit is not None else [])
        final_output = outputs

    if len(mel_chunks) <= 0:
        mel_pred = final_output["mel_out"][0]
    else:
        mel_pred = torch.cat(mel_chunks, dim=0)
    return StreamingEvalResult(
        mel_pred=mel_pred,
        final_output=final_output,
        mel_lengths=mel_lengths,
        commit_history=commit_history,
    )
