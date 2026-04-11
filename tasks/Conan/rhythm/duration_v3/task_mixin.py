from __future__ import annotations

from tasks.Conan.rhythm.duration_v3.targets import DurationV3TargetBuildConfig, build_duration_v3_loss_targets
from tasks.Conan.rhythm.duration_v3.task_config import is_duration_v3_prompt_summary_backbone
from utils.commons.hparams import hparams


class DurationV3TaskMixin:
    def _get_rhythm_v3_baseline_module(self):
        if self.model is None:
            return None
        rhythm_frontend = getattr(self.model, "rhythm_unit_frontend", None)
        if rhythm_frontend is None:
            return None
        getter = getattr(rhythm_frontend, "get_baseline_module", None)
        if callable(getter):
            return getter()
        return getattr(rhythm_frontend, "baseline", None)

    @staticmethod
    def _resolve_rhythm_v3_baseline_train_mode() -> str:
        return str(hparams.get("rhythm_v3_baseline_train_mode", "joint") or "joint").strip().lower()

    @classmethod
    def _is_rhythm_v3_baseline_pretrain_mode(cls) -> bool:
        return cls._resolve_rhythm_v3_baseline_train_mode() == "pretrain"

    def _collect_rhythm_v3_baseline_only_params(self):
        if self.model is None or not getattr(self.model, "rhythm_enable_v3", False):
            return []
        if is_duration_v3_prompt_summary_backbone(hparams.get("rhythm_v3_backbone", "global_only")):
            return []
        baseline_module = self._get_rhythm_v3_baseline_module()
        if baseline_module is None:
            return []
        return self._task_runtime_support().dedup_trainable_params(list(baseline_module.parameters()))

    @classmethod
    def _should_collect_rhythm_v3_baseline_params(cls) -> bool:
        if is_duration_v3_prompt_summary_backbone(hparams.get("rhythm_v3_backbone", "global_only")):
            return False
        baseline_mode = cls._resolve_rhythm_v3_baseline_train_mode()
        if baseline_mode != "joint":
            return False
        if bool(hparams.get("rhythm_v3_freeze_baseline", False)):
            return False
        return True

    def _build_duration_v3_target_build_config(self) -> DurationV3TargetBuildConfig:
        return self._task_runtime_support().build_duration_v3_target_build_config()

    def _build_duration_v3_loss_targets(self, output, sample):
        return build_duration_v3_loss_targets(
            sample=sample,
            output=output,
            config=self._build_duration_v3_target_build_config(),
        )


__all__ = ["DurationV3TaskMixin"]
