# Rhythm Weak-Factorization Mainline

## Positioning

The maintained rhythm path should be described as:

- **projector-grounded weak factorization**
- **explicit hierarchical rhythm planning**
- **role-constrained planner surfaces**

It should **not** be described as a fully identifiable disentangled factor model.

## Maintained compact contract

### Compact conditioning

- `global_rate`
- `pause_ratio`
- `local_rate_trace`
- `boundary_trace`
- `source_boundary_cue`

### Planner surfaces

- `speech_budget_win`
- `pause_budget_win`
- `dur_shape_unit` (`dur_logratio_unit` runtime alias)
- `pause_shape_unit` (`pause_weight_unit` runtime alias)
- `boundary_score_unit` (deterministic sidecar, not a free latent)

### Binding execution contract

- `speech_exec`
- `pause_exec`
- `commit_frontier`
- `next_state`

## Maintained implementation choices

1. **No learnable `boundary_latent` in the mainline**
   - boundary evidence is deterministic:
     - source-side prefix-safe cue
     - reference-side boundary trace
   - this reduces role competition with pause placement

2. **No budget-hidden leakage into redistribution**
   - redistribution sees only local unit state + compact trace/boundary evidence
   - budget policy stays global

3. **Shape-aware distillation**
   - `distill_speech_shape`
   - `distill_pause_shape`
   - keep execution/budget/prefix supervision separate from shape supervision

4. **Projector is the only binding execution authority**
   - planner surfaces are explicit and debuggable
   - executed speech/pause and monotonic state are the maintained truth

## Evaluation added for this mainline

### Factor-wise intervention

Use `scripts/eval_rhythm_factor_intervention.py`.

Purpose:

- perturb only the compact planner-facing conditioning
- inspect which planner/execution surfaces actually move
- expose leakage instead of inferring it from MOS alone

Recommended checks:

- `global_rate_up` should mainly move `speech_budget_win`
- `pause_ratio_up` should mainly move `pause_budget_win`
- `local_rate_shape_up` should mainly move `dur_shape_unit`
- `boundary_trace_up` should mainly move `pause_shape_unit` / `boundary_score_unit` / `commit_frontier`

### Seed / checkpoint stability

Use `scripts/eval_rhythm_seed_stability.py`.

Purpose:

- compare compact planner surfaces across seeds or checkpoints
- detect “same execution, different internal surface” failure modes

Interpretation:

- low execution drift + high planner-surface drift means the system is still compensating internally
- lower pairwise drift on `speech_budget_win`, `pause_budget_win`, `dur_shape_unit`, `pause_shape_unit` is better

## Example commands

```bash
python scripts/eval_rhythm_factor_intervention.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --binary_data_dir data/binary/libritts_local_real_smoke_rhythm_v5 \
  --processed_data_dir data/processed/libritts_local_real_smoke \
  --ckpt checkpoints/your_ckpt.ckpt \
  --split valid \
  --max_items 4 \
  --output_json artifacts/rhythm_factor_intervention.json
```

```bash
python scripts/eval_rhythm_seed_stability.py \
  --config egs/conan_emformer_rhythm_v2_student_kd.yaml \
  --binary_data_dir data/binary/libritts_local_real_smoke_rhythm_v5 \
  --processed_data_dir data/processed/libritts_local_real_smoke \
  --ckpts checkpoints/run_a.ckpt checkpoints/run_b.ckpt checkpoints/run_c.ckpt \
  --split valid \
  --max_items 4 \
  --output_json artifacts/rhythm_seed_stability.json
```

## Practical rule

Keep adding **evidence**, not more free variables.

If a new head does not have:

- a clear compact input contract
- a role-specific supervision path
- an intervention protocol
- a stability check

it probably does not belong on the maintained mainline.
