from utils.commons.single_thread_env import maybe_apply_single_thread_env_from_env

maybe_apply_single_thread_env_from_env()

from utils.commons.hparams import hparams, set_hparams
import importlib


def resolve_task_cls(task_cls_path: str):
    task_cls_path = str(task_cls_path or "").strip()
    if not task_cls_path:
        raise ValueError("task_cls must be a non-empty import path.")
    if "." not in task_cls_path:
        raise ValueError("task_cls must look like 'pkg.module.ClassName'.")
    pkg, cls_name = task_cls_path.rsplit(".", 1)
    if not pkg or not cls_name:
        raise ValueError("task_cls must look like 'pkg.module.ClassName'.")
    try:
        module = importlib.import_module(pkg)
    except Exception as exc:
        raise ImportError(f"Failed to import task module '{pkg}' for task_cls='{task_cls_path}'.") from exc
    try:
        task_cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise ImportError(f"Task class '{cls_name}' is not defined in module '{pkg}'.") from exc
    return task_cls


def run_task(task_cls_path: str | None = None):
    task_cls = resolve_task_cls(task_cls_path or hparams.get("task_cls", ""))
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    run_task()
