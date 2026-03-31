import torch
import numpy as np
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from modules.Conan.rhythm.factory import build_streaming_rhythm_module_from_hparams
from modules.Conan.rhythm.supervision import build_reference_guided_targets, build_reference_teacher_targets
from modules.Conan.rhythm.unit_frontend import RhythmUnitFrontend


if __name__ == '__main__':
    hparams = {
        'content_embedding_dim': 128,
        'rhythm_hidden_size': 32,
        'rhythm_trace_bins': 12,
        'rhythm_stats_dim': 6,
        'rhythm_trace_dim': 5,
        'rhythm_max_total_logratio': 0.8,
        'rhythm_max_unit_logratio': 0.6,
        'rhythm_pause_share_max': 0.45,
        'rhythm_projector_min_speech_frames': 1.0,
        'rhythm_projector_max_speech_expand': 3.0,
        'rhythm_projector_tail_hold_units': 2,
        'rhythm_projector_boundary_commit_threshold': 0.45,
    }
    frontend = RhythmUnitFrontend(silent_token=57, separator_aware=True)
    batch = frontend.from_token_lists([
        [1, 1, 1, 2, 2, 57, 3, 3, 4, 4],
        [5, 5, 6, 6, 6, 7, 57, 8, 8, 8],
    ], device=torch.device('cpu'))
    model = build_streaming_rhythm_module_from_hparams(hparams)
    out = model(
        content_units=batch.content_units,
        dur_anchor_src=batch.dur_anchor_src,
        unit_mask=batch.unit_mask,
        open_run_mask=batch.open_run_mask,
        ref_mel=torch.randn(2, 80, 64),
    )
    guidance = build_reference_guided_targets(
        dur_anchor_src=batch.dur_anchor_src[0].cpu().numpy(),
        unit_mask=batch.unit_mask[0].cpu().numpy(),
        ref_rhythm_stats=np.array([0.15, 2.0, 3.0, 0.0, 0.2, 0.8], dtype=np.float32),
        ref_rhythm_trace=np.random.randn(12, 5).astype(np.float32),
    )
    teacher = build_reference_teacher_targets(
        dur_anchor_src=batch.dur_anchor_src[0].cpu().numpy(),
        unit_mask=batch.unit_mask[0].cpu().numpy(),
        ref_rhythm_stats=np.array([0.15, 2.0, 3.0, 0.0, 0.2, 0.8], dtype=np.float32),
        ref_rhythm_trace=np.random.randn(12, 5).astype(np.float32),
    )
    print('speech_exec shape:', tuple(out.speech_duration_exec.shape))
    print('pause_exec shape:', tuple(out.pause_after_exec.shape))
    print('commit_frontier:', out.commit_frontier.tolist())
    print('guidance keys:', sorted(guidance.keys()))
    print('teacher keys:', sorted(teacher.keys()))
