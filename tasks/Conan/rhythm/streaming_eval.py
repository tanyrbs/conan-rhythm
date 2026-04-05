from __future__ import annotations

from dataclasses import dataclass

import torch

from .streaming_commit import extract_incremental_committed_mel


@dataclass
class StreamingEvalResult:
    mel_pred: torch.Tensor
    final_output: dict
    mel_lengths: list[int]
    committed_mel_lengths: list[int]
    commit_history: list[list[int]]
    prefix_exec_deltas: list[float]
    backlog_history: list[float]
    clock_history: list[float]
    blank_ratio_history: list[float]


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
    committed_mel_lengths = []
    commit_history = []
    prefix_exec_deltas = []
    backlog_history = []
    clock_history = []
    blank_ratio_history = []
    prev_committed_len = 0
    prev_commit_units = 0
    prev_speech_exec = None
    prev_pause_exec = None
    rhythm_state = None
    rhythm_ref_conditioning = None
    rhythm_unitizer_state = None
    final_output = None
    rhythm_frontend = getattr(getattr(task, "model", None), "rhythm_unit_frontend", None)

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * tokens_per_chunk
        end_idx = min((chunk_idx + 1) * tokens_per_chunk, total_tokens)
        sample_chunk = {k: v for k, v in sample.items()}
        chunk_tokens = content_full[:, start_idx:end_idx]
        chunk_len = int(chunk_tokens.size(1))
        sample_chunk["content"] = chunk_tokens
        sample_chunk["mel_lengths"] = torch.tensor([chunk_len], dtype=torch.long, device=content_full.device)
        sample_chunk["content_lengths"] = torch.tensor([chunk_len], dtype=torch.long, device=content_full.device)
        if rhythm_frontend is not None:
            if rhythm_unitizer_state is None:
                rhythm_unitizer_state = rhythm_frontend.init_stream_state(batch_size=1, device=content_full.device)
            unit_batch, rhythm_unitizer_state = rhythm_frontend.step_content_tensor(
                chunk_tokens,
                state=rhythm_unitizer_state,
                content_lengths=sample_chunk["content_lengths"],
                mark_last_open=True,
            )
            sample_chunk["content_units"] = unit_batch.content_units
            sample_chunk["dur_anchor_src"] = unit_batch.dur_anchor_src
            sample_chunk["open_run_mask"] = unit_batch.open_run_mask
            sample_chunk["sealed_mask"] = unit_batch.sealed_mask
            sample_chunk["sep_hint"] = unit_batch.sep_hint
            sample_chunk["boundary_confidence"] = unit_batch.boundary_confidence
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
        mel_new, prev_committed_len = extract_incremental_committed_mel(
            outputs,
            prev_committed_len=prev_committed_len,
            batch_index=0,
        )
        if mel_new.numel() > 0:
            mel_chunks.append(mel_new)
        mel_lengths.append(int(mel_out.shape[0]))
        committed_mel_lengths.append(int(prev_committed_len))
        commit = outputs.get("commit_frontier")
        commit_list = commit.detach().cpu().tolist() if commit is not None else []
        commit_history.append(commit_list)
        execution = outputs.get("rhythm_execution")
        if execution is not None and prev_speech_exec is not None and prev_pause_exec is not None and prev_commit_units > 0:
            curr_speech = execution.speech_duration_exec[0, :prev_commit_units]
            curr_pause = getattr(execution, "blank_duration_exec", execution.pause_after_exec)[0, :prev_commit_units]
            delta = torch.cat(
                [
                    (curr_speech - prev_speech_exec[0, :prev_commit_units]).abs(),
                    (curr_pause - prev_pause_exec[0, :prev_commit_units]).abs(),
                ],
                dim=0,
            ).max().item()
            prefix_exec_deltas.append(float(delta))
        else:
            prefix_exec_deltas.append(0.0)
        if execution is not None:
            prev_speech_exec = execution.speech_duration_exec.detach()
            prev_pause_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec).detach()
            unit_batch = outputs.get("rhythm_unit_batch")
            if unit_batch is not None:
                unit_mask = unit_batch.unit_mask[0].float()
                visible = unit_mask.sum().clamp_min(1.0)
                blank_exec = getattr(execution, "blank_duration_exec", execution.pause_after_exec)[0]
                blank_ratio_history.append(float(((blank_exec > 0.5).float() * unit_mask).sum().item() / visible.item()))
            else:
                blank_ratio_history.append(0.0)
        prev_commit_units = int(commit_list[0]) if len(commit_list) > 0 else 0
        state_next = outputs.get("rhythm_state_next")
        if state_next is not None:
            backlog_history.append(float(state_next.backlog[0].item()))
            clock_history.append(float(state_next.clock_delta[0].item()))
        else:
            backlog_history.append(0.0)
            clock_history.append(0.0)
        final_output = outputs

    final_mel = final_output["mel_out"][0]
    if prev_committed_len < int(final_mel.shape[0]):
        mel_chunks.append(final_mel[prev_committed_len:])
    if len(mel_chunks) <= 0:
        mel_pred = final_mel
    else:
        mel_pred = torch.cat(mel_chunks, dim=0)
    return StreamingEvalResult(
        mel_pred=mel_pred,
        final_output=final_output,
        mel_lengths=mel_lengths,
        committed_mel_lengths=committed_mel_lengths,
        commit_history=commit_history,
        prefix_exec_deltas=prefix_exec_deltas,
        backlog_history=backlog_history,
        clock_history=clock_history,
        blank_ratio_history=blank_ratio_history,
    )
