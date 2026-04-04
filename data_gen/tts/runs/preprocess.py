from utils.commons.single_thread_env import maybe_apply_single_thread_env_from_env
from utils.commons.hparams import hparams, set_hparams
import importlib

maybe_apply_single_thread_env_from_env()


def preprocess():
    assert hparams['preprocess_cls'] != ''

    pkg = ".".join(hparams["preprocess_cls"].split(".")[:-1])
    cls_name = hparams["preprocess_cls"].split(".")[-1]
    process_cls = getattr(importlib.import_module(pkg), cls_name)
    process_cls().process()


if __name__ == '__main__':
    set_hparams()
    preprocess()
