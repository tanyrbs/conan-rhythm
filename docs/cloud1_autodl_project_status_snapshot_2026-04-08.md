> **Archive snapshot only (legacy v2 teacher_offline run).**
> This file records a historical training/status snapshot and may reflect runs
> started before the current repository HEAD. It is not the live
> runtime-contract or architecture document. For the current `rhythm_v3`
> mainline semantics, use:
>
> - `README.md`
> - `docs/rhythm_migration_plan.md`
>
> `docs/autodl_training_handoff.md` is now legacy v2 operational guidance only.

# Archive Project Status (legacy v2 snapshot)

鏇存柊鏃堕棿锛?026-04-08 UTC

## 1. 褰撳墠椤圭洰鐘舵€?

### 褰撳墠鍞竴缁存姢涓荤嚎
- stage: `teacher_offline`
- data: `mixed train100 + train360`
- lineage: `v6`
- pause path: `split-head support + allocation`
- semantic: `weight-only warm-start`

### 褰撳墠 active experiment
- exp: `teacher_offline_train100_360_v6_split_heads_restart17500_fix1`
- log: `logs/teacher_offline_train100_360_v6_split_heads_restart17500_fix1.log`
- ckpt dir: `checkpoints/teacher_offline_train100_360_v6_split_heads_restart17500_fix1`
- bootstrap: `/root/autodl-tmp/project/conan-rhythm/checkpoints/model_ckpt_steps_17500.ckpt`

### 褰撳墠杩愯鐘舵€?
- 璁粌杩涚▼浠嶅湪杩愯
- latest observed train step token in log: **33236**
- 宸插畬鎴?validation step锛歚5000 / 10000 / 15000 / 20000 / 25000 / 30000`

> 娉ㄦ剰锛氬綋鍓嶆鍦ㄨ繍琛岀殑杩涚▼鏃╀簬鏈€杩戜竴娆′唬鐮佹彁浜ゅ惎鍔紝鍥犳瀹冨弽鏄犵殑鏄?*鏃х殑杩涚▼鍐呬唬鐮?*锛屼笉鏄綋鍓嶄粨搴?HEAD 瀵瑰簲鐨勬柊閫昏緫銆?

---

## 2. 褰撳墠璁粌杩涘害璁板綍

| val step | pause_event_f1 | exec_total_corr | prefix_drift_l1 | support_cover_at_topk | recall_drop_post_from_planner | boundary recall |
|---|---:|---:|---:|---:|---:|---:|
| 5000  | 0.8200 | 0.9192 | 21.2524 | 0.9357 | 0.0380 | 0.0019 |
| 10000 | 0.8114 | 0.9171 | 22.9551 | 0.9307 | 0.0356 | 0.0019 |
| 15000 | 0.8215 | 0.9172 | 21.4709 | 0.9344 | 0.0460 | 0.0018 |
| 20000 | 0.8268 | 0.9180 | 21.3356 | 0.9366 | 0.0491 | 0.0019 |
| 25000 | 0.8226 | 0.9203 | 21.4935 | 0.9367 | 0.0474 | 0.0019 |
| 30000 | 0.8271 | 0.9185 | 21.8889 | 0.9355 | 0.0531 | 0.0019 |

### 褰撳墠 best 璁板綍
- overall best (`model_ckpt_best.pt`): **@15000**锛屾寜 `val_loss=0.22422`
- pause best (`model_ckpt_pause_best.pt`): **@30000**锛屾寜 `rhythm_metric_pause_event_f1=0.82708`
- 鏈€杩?step ckpt锛歚20000 / 25000 / 30000`

---

## 3. 褰撳墠鍒ゆ柇

### 缁撹
**鍏ㄥ眬鍋ュ悍锛屼絾 boundary 浠嶆槑鏄炬粸鍚庛€?*

### 鏀寔杩欎釜鍒ゆ柇鐨勪簨瀹?
- `pause_event_f1` 鏁翠綋缁存寔鍦?`0.81~0.83` 鍖洪棿锛屾病鏈変富绾垮穿鍧忚抗璞?
- `exec_total_corr` 缁存寔楂樹綅锛宍@25000=0.9203`銆乣@30000=0.9185`
- `prefix_drift_l1` 缁存寔鍦?`21~22` 鍖洪棿锛宍@30000` 鐣ユ湁涓婅浣嗘湭澶辨帶
- `pause_support_cover_at_topk` 缁存寔鍦?`0.93+`锛岃鏄?planner 鈫?projector 鐨勪富瑕?support 瑕嗙洊浠嶅仴搴?
- `pause_recall_drop_post_from_planner` 绾?`0.04~0.05`锛宍@30000` 鐣ユ姮鍒?`0.0531`锛屼粛闇€鐩揣
- `boundary recall` 鎸佺画绾?`0.0018~0.0019`锛岃鏄?boundary 瀛愰泦鏀剁泭灏氭湭浣撶幇鍑烘潵

### 褰撳墠鏈€閲嶈椋庨櫓
褰撳墠 Stage1 杩樹笉鑳藉洜涓哄叏灞€鎸囨爣鍋ュ悍灏辩洿鎺ュ垽瀹氬畬鎴愶紝涓昏鍗＄偣浠嶆槸锛?
- boundary pause 瀛愰泦鍑犱箮娌℃湁鎶捣鏉?
- boundary 鐩稿叧鏀剁泭杩樻病鏈夌ǔ瀹氬弽鏄犲埌鐩戞帶鍊间笂

---

## 4. 褰撳墠浠ｇ爜涓庤缁冪殑鍏崇郴

褰撳墠浠撳簱 HEAD 宸茬粡鍖呭惈涓€鎵逛唬鐮佷笌鏂囨。淇锛屼富瑕佸寘鎷細
- 鏂囨。涓荤嚎鏀舵暃
- commit controller / projector / metrics / losses 鐨勮嫢骞蹭慨澶?
- boundary valid-only 鎸囨爣涓庣洃鎺цˉ寮?

浣嗚繖浜涗慨鏀?*灏氭湭搴旂敤鍒板綋鍓嶆鍦ㄨ繍琛岀殑璁粌杩涚▼**銆傚洜姝わ細
- 褰撳墠 run 浠嶅彲浣滀负鏃ч€昏緫 baseline 瑙傚療
- 濡傛灉瑕侀獙璇佹柊閫昏緫锛屽繀椤绘柊寮€ experiment 鎴栭噸鍚?run

---

## 5. 褰撳墠涓嬩竴姝ュ缓璁?

1. 淇濇寔褰撳墠 run 缁х画璺戝埌涓嬩竴涓?validation 鐐?
2. 鏂囨。鍙繚鐣?README + code/config guide + project status 涓変唤
3. 鏈湴浠ｇ爜鏁寸悊鍚庢彁浜ゅ埌 `autodl` 鍒嗘敮
4. 闇€瑕侀獙璇佹柊 boundary 淇鏃讹紝浣跨敤 `17500` warm-start 鏂板紑 experiment锛屼笉瑕佹薄鏌撳綋鍓?run
