# Archive Project Status (legacy v2 snapshot, 2026-04-08)

> Archive snapshot only.
> This file records a historical v2 `teacher_offline` run and is **not** the
> current architecture or runtime-contract document.
>
> Current docs:
>
> - `README.md`
> - `docs/rhythm_migration_plan.md`
> - `inference/README.md`

## Preserved snapshot identity

Historical run:

- stage: `teacher_offline`
- data: mixed `train100 + train360`
- lineage: `v6`
- active experiment:
  - `teacher_offline_train100_360_v6_split_heads_restart17500_fix1`
- log:
  - `logs/teacher_offline_train100_360_v6_split_heads_restart17500_fix1.log`
- checkpoint dir:
  - `checkpoints/teacher_offline_train100_360_v6_split_heads_restart17500_fix1`
- warm-start checkpoint:
  - `/root/autodl-tmp/project/conan-rhythm/checkpoints/model_ckpt_steps_17500.ckpt`

## Preserved validation table

| val step | pause_event_f1 | exec_total_corr | prefix_drift_l1 | support_cover_at_topk | recall_drop_post_from_planner | boundary_recall |
|---|---:|---:|---:|---:|---:|---:|
| 5000  | 0.8200 | 0.9192 | 21.2524 | 0.9357 | 0.0380 | 0.0019 |
| 10000 | 0.8114 | 0.9171 | 22.9551 | 0.9307 | 0.0356 | 0.0019 |
| 15000 | 0.8215 | 0.9172 | 21.4709 | 0.9344 | 0.0460 | 0.0018 |
| 20000 | 0.8268 | 0.9180 | 21.3356 | 0.9366 | 0.0491 | 0.0019 |
| 25000 | 0.8226 | 0.9203 | 21.4935 | 0.9367 | 0.0474 | 0.0019 |
| 30000 | 0.8271 | 0.9185 | 21.8889 | 0.9355 | 0.0531 | 0.0019 |

## Preserved checkpoint notes

- overall best (`model_ckpt_best.pt`): `@15000`
- pause best (`model_ckpt_pause_best.pt`): `@30000`
- recent step checkpoints kept in the original run included:
  - `20000`
  - `25000`
  - `30000`

## Archive rule

Keep this file as a frozen historical note only.
Do not extend it into a current-branch architecture document, and do not use it
as the authority for `rhythm_v3` design decisions.
