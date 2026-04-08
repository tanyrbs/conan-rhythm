import random
import shutil
import traceback
from datetime import datetime

try:
    from torch.amp import GradScaler
    _TORCH_AMP_NEW_API = True
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler
    _TORCH_AMP_NEW_API = False
import numpy as np
import torch.optim
import torch.utils.data
import copy
import inspect
import logging
import os
import re
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.commons.ckpt_utils import get_last_checkpoint, get_all_ckpts
from utils.commons.ddp_config import (
    build_ddp_auto_signature,
    get_ddp_auto_min_step,
    load_ddp_logging_data,
    resolve_ddp_runtime_config,
    save_ddp_logging_data,
    select_ddp_logging_hint,
)
from utils.commons.ddp_utils import DDP
from utils.commons.hparams import hparams
from utils.commons.trainer_loop import TrainerLoopMixin
from utils.os_utils import remove_file


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class Trainer(TrainerLoopMixin):
    def __init__(
            self,
            work_dir,
            default_save_path=None,
            accumulate_grad_batches=1,
            max_updates=160000,
            print_nan_grads=False,
            val_check_interval=2000,
            num_sanity_val_steps=5,
            amp=False,
            # tb logger
            log_save_interval=100,
            tb_log_interval=10,
            # checkpoint
            monitor_key='val_loss',
            monitor_mode='min',
            num_ckpt_keep=5,
            save_best=True,
            extra_monitor_key='',
            extra_monitor_mode='max',
            extra_monitor_filename='model_ckpt_pause_best.pt',
            resume_from_checkpoint=0,
            seed=1234,
            debug=False,
    ):
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_updates = max_updates
        if self.accumulate_grad_batches > 1:
            logging.warning(
                "Trainer global_step/max_updates/val_check_interval are micro-batch based while "
                "accumulate_grad_batches > 1. Plan schedules and checkpoint cadence accordingly."
            )
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.default_save_path = default_save_path
        self.resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint > 0 else None
        self.seed = seed
        self.debug = debug
        # model and optm
        self.task = None
        self.optimizers = []

        # trainer state
        self.testing = False
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0
        self.last_saved_ckpt_step = None

        # configure checkpoint
        self.monitor_key = monitor_key
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best
        self.monitor_op = np.less if monitor_mode == 'min' else np.greater
        self.best_val_results = np.Inf if monitor_mode == 'min' else -np.Inf
        self.mode = monitor_mode
        self.extra_monitor_key = str(extra_monitor_key or '').strip()
        self.extra_monitor_filename = str(extra_monitor_filename or 'model_ckpt_pause_best.pt').strip()
        if self.extra_monitor_key:
            self.extra_monitor_op = np.less if extra_monitor_mode == 'min' else np.greater
            self.extra_monitor_best = np.Inf if extra_monitor_mode == 'min' else -np.Inf
            self.extra_monitor_mode = extra_monitor_mode
        else:
            self.extra_monitor_op = None
            self.extra_monitor_best = None
            self.extra_monitor_mode = extra_monitor_mode

        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_tokens = [x.strip() for x in visible_env.split(",") if x.strip() != ""]
        if len(visible_tokens) > 0:
            self.all_gpu_ids = list(range(len(visible_tokens)))
        elif torch.cuda.is_available():
            self.all_gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.all_gpu_ids = []
        self.num_gpus = len(self.all_gpu_ids)
        self.on_gpu = self.num_gpus > 0 and torch.cuda.is_available()
        self.root_gpu = 0
        logging.info(f'GPU available: {torch.cuda.is_available()}, GPU used: {self.all_gpu_ids}')
        self.use_ddp = self.num_gpus > 1
        self.proc_rank = 0
        # Tensorboard logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.tb_log_interval = tb_log_interval
        self.amp = amp
        self.autocast_enabled = bool(self.amp and self.on_gpu and torch.cuda.is_available())
        scaler_enabled = self.autocast_enabled
        if _TORCH_AMP_NEW_API:
            self.amp_scalar = GradScaler(device='cuda', enabled=scaler_enabled)
        else:  # pragma: no cover
            self.amp_scalar = GradScaler(enabled=scaler_enabled)

    def test(self, task_cls):
        self.testing = True
        self.fit(task_cls)

    def fit(self, task_cls):
        if self.use_ddp:
            mp.spawn(self.ddp_run, nprocs=self.num_gpus, args=(task_cls, copy.deepcopy(hparams)))
        else:
            self.task = task_cls()
            self.task.trainer = self
            self.run_single_process(self.task)
        return 1

    def ddp_run(self, gpu_idx, task_cls, hparams_):
        hparams.update(hparams_)
        self.proc_rank = gpu_idx
        self.init_ddp_connection(self.proc_rank, self.num_gpus)
        if dist.get_rank() != 0 and not self.debug:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        task = task_cls()
        task.trainer = self
        torch.cuda.set_device(gpu_idx)
        self.root_gpu = gpu_idx
        self.task = task
        self.run_single_process(task)

    def run_single_process(self, task):
        """Sanity check a few things before starting actual training.

        :param task:
        """
        # build model, optm and load checkpoint
        if self.proc_rank == 0:
            self.save_terminal_logs()
            if not self.testing:
                self.save_codes()

        model = task.build_model()
        if model is not None:
            task.model = model
        checkpoint, _ = get_last_checkpoint(self.work_dir, self.resume_from_checkpoint)
        if checkpoint is not None:
            self.restore_weights(checkpoint)
        elif self.on_gpu:
            task.cuda(self.root_gpu)
        if not self.testing:
            self.optimizers = task.configure_optimizers()
            self.fisrt_epoch = True
        if checkpoint is not None:
            self.restore_opt_state(checkpoint)
        del checkpoint
        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        if self.use_ddp:
            self.task = self.configure_ddp(self.task)
            dist.barrier()

        task_ref = self.get_task_ref()
        task_ref.trainer = self
        task_ref.testing = self.testing
        # link up experiment object
        if self.proc_rank == 0:
            task_ref.build_tensorboard(save_dir=self.work_dir, name='tb_logs')
        else:
            os.makedirs('tmp', exist_ok=True)
            task_ref.build_tensorboard(save_dir='tmp', name='tb_tmp')
        self.logger = task_ref.logger
        try:
            if self.testing:
                self.run_evaluation(test=True)
            else:
                self.train()
                self.maybe_save_ddp_logging_data()
        except KeyboardInterrupt as e:
            traceback.print_exc()
            task_ref.on_keyboard_interrupt()

    ####################
    # valid and test
    ####################
    def run_evaluation(self, test=False):
        return TrainerLoopMixin.run_evaluation(self, test=test)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        return TrainerLoopMixin.evaluate(
            self,
            task,
            test=test,
            tqdm_desc=tqdm_desc,
            max_batches=max_batches,
        )

    ####################
    # train
    ####################
    def train(self):
        return TrainerLoopMixin.train(self)

    def run_training_batch(self, batch_idx, batch):
        return TrainerLoopMixin.run_training_batch(self, batch_idx, batch)

    ####################
    # load and save checkpoint
    ####################
    def restore_weights(self, checkpoint):
        # load model state
        task_ref = self.get_task_ref()

        for k, v in checkpoint['state_dict'].items():
            getattr(task_ref, k).load_state_dict(v)

        if self.on_gpu:
            task_ref.cuda(self.root_gpu)
        # load training state (affects trainer only)
        self.best_val_results = checkpoint.get('checkpoint_callback_best', self.best_val_results)
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.last_saved_ckpt_step = self.global_step
        task_ref.global_step = self.global_step
        extra_monitor_state = checkpoint.get('extra_monitor_state', None)
        if (
                isinstance(extra_monitor_state, dict)
                and self.extra_monitor_key
                and str(extra_monitor_state.get('key', '')) == self.extra_monitor_key
        ):
            restored_best = extra_monitor_state.get('best', None)
            if restored_best is not None:
                self.extra_monitor_best = float(restored_best)

        # wait for all models to restore weights
        if self.use_ddp:
            # wait for all processes to catch up
            dist.barrier()

    def restore_opt_state(self, checkpoint):
        if self.testing:
            return
        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if optimizer is None:
                return
            try:
                optimizer.load_state_dict(opt_state)
                # move optimizer to GPU 1 weight at a time
                if self.on_gpu:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.root_gpu)
            except ValueError:
                print("| WARMING: optimizer parameters not match !!!")
        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            print(e)
            return
        did_restore = True
        return did_restore

    def save_checkpoint(self, epoch, logs=None):
        ckpt_path = f'{self.work_dir}/model_ckpt_steps_{self.global_step}.ckpt'
        logging.info(f'Epoch {epoch:05d}@{self.global_step}: saving model to {ckpt_path}')
        self._atomic_save(ckpt_path)
        self.last_saved_ckpt_step = self.global_step
        self.maybe_save_ddp_logging_data()
        for old_ckpt in get_all_ckpts(self.work_dir)[self.num_ckpt_keep:]:
            remove_file(old_ckpt)
            logging.info(f'Delete ckpt: {os.path.basename(old_ckpt)}')
        if self.save_best:
            self._maybe_save_best_alias(
                epoch=epoch,
                logs=logs,
                monitor_key=self.monitor_key,
                monitor_op=self.monitor_op,
                current_best_attr='best_val_results',
                filename='model_ckpt_best.pt',
            )
            if self.extra_monitor_key and self.extra_monitor_op is not None:
                self._maybe_save_best_alias(
                    epoch=epoch,
                    logs=logs,
                    monitor_key=self.extra_monitor_key,
                    monitor_op=self.extra_monitor_op,
                    current_best_attr='extra_monitor_best',
                    filename=self.extra_monitor_filename,
                )

    @staticmethod
    def _resolve_logged_metric(logs, monitor_key):
        if logs is None or not monitor_key:
            return None
        if monitor_key in logs and isinstance(logs.get(monitor_key), (int, float, np.number)):
            return float(logs[monitor_key])
        tb_log = logs.get('tb_log', None)
        if isinstance(tb_log, dict):
            if monitor_key in tb_log and isinstance(tb_log.get(monitor_key), (int, float, np.number)):
                return float(tb_log[monitor_key])
            val_key = f'val/{monitor_key}'
            if val_key in tb_log and isinstance(tb_log.get(val_key), (int, float, np.number)):
                return float(tb_log[val_key])
        return None

    def _maybe_save_best_alias(
            self,
            *,
            epoch,
            logs,
            monitor_key,
            monitor_op,
            current_best_attr,
            filename,
    ):
        current = self._resolve_logged_metric(logs, monitor_key)
        if current is None:
            return
        best_so_far = getattr(self, current_best_attr)
        if best_so_far is None or monitor_op(current, best_so_far):
            setattr(self, current_best_attr, current)
            best_filepath = f'{self.work_dir}/{filename}'
            logging.info(
                f'Epoch {epoch:05d}@{self.global_step}: {monitor_key} reached {current:0.5f}. '
                f'Saving model to {best_filepath}')
            self._atomic_save(best_filepath)

    def _atomic_save(self, filepath):
        checkpoint = self.dump_checkpoint()
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, filepath)

    def dump_checkpoint(self):
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step,
                      'checkpoint_callback_best': self.best_val_results}
        if self.extra_monitor_key:
            checkpoint['extra_monitor_state'] = {
                'key': self.extra_monitor_key,
                'mode': self.extra_monitor_mode,
                'best': self.extra_monitor_best,
                'filename': self.extra_monitor_filename,
            }
        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states
        task_ref = self.get_task_ref()
        checkpoint['state_dict'] = {
            k: v.state_dict() for k, v in task_ref.named_children() if len(list(v.parameters())) > 0}
        return checkpoint

    ####################
    # DDP
    ####################
    def _ddp_logging_data_path(self):
        return os.path.join(self.work_dir, 'ddp_logging_data.json')

    def _load_ddp_logging_hint(self):
        return load_ddp_logging_data(self._ddp_logging_data_path())

    def _build_ddp_auto_signature(self, task):
        return build_ddp_auto_signature(hparams=hparams, task=task)

    def maybe_save_ddp_logging_data(self):
        if self.proc_rank != 0 or not isinstance(self.task, DDP):
            return
        if not hasattr(self.task, '_get_ddp_logging_data'):
            return
        try:
            logging_data = self.task._get_ddp_logging_data()
        except Exception as exc:  # pragma: no cover
            logging.warning('Failed to query DDP logging data: %s', exc)
            return
        save_ddp_logging_data(
            self._ddp_logging_data_path(),
            logging_data,
            global_step=self.global_step,
            epoch=self.current_epoch,
            signature=self._build_ddp_auto_signature(self.task.module),
        )

    def configure_ddp(self, task):
        ddp_signature = self._build_ddp_auto_signature(task)
        ddp_logging_data = select_ddp_logging_hint(
            self._load_ddp_logging_hint(),
            signature=ddp_signature,
            min_saved_global_step=get_ddp_auto_min_step(hparams),
        )
        find_unused, static_graph = resolve_ddp_runtime_config(
            hparams.get('ddp_find_unused_parameters', 'auto'),
            hparams.get('ddp_static_graph', 'auto'),
            ddp_logging_data=ddp_logging_data,
            default_find_unused=True,
            default_static_graph=False,
        )
        ddp_kwargs = {
            'device_ids': [self.root_gpu],
            'find_unused_parameters': find_unused,
        }
        supports_static_graph = 'static_graph' in inspect.signature(DDP.__init__).parameters
        if supports_static_graph:
            ddp_kwargs['static_graph'] = static_graph
        elif static_graph:
            logging.warning(
                'DDP static_graph resolved true but current DistributedDataParallel does not expose static_graph.'
            )
        logging.info(
            'DDP config resolved: find_unused_parameters=%s, static_graph=%s, can_set_static_graph=%s, auto_hint_loaded=%s',
            find_unused,
            static_graph,
            bool(ddp_logging_data.get('can_set_static_graph', False)),
            bool(ddp_logging_data),
        )
        task = DDP(task, **ddp_kwargs)
        random.seed(self.seed)
        np.random.seed(self.seed)
        return task

    def init_ddp_connection(self, proc_rank, world_size):
        root_node = '127.0.0.1'
        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]
            number = re.sub('[^0-9]', '', number)
            root_node = name + number
        return root_node

    ####################
    # utils
    ####################
    def get_task_ref(self):
        from utils.commons.base_task import BaseTask
        task: BaseTask = self.task.module if isinstance(self.task, DDP) else self.task
        return task

    def log_metrics_to_tb(self, metrics, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        """
        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0:
            self.log_metrics(self.logger, scalar_metrics, step=step)

    @staticmethod
    def log_metrics(logger, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def save_terminal_logs(self):
        t = datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs(f'{self.work_dir}/terminal_logs', exist_ok=True)
        # Tee(f'{self.work_dir}/terminal_logs/log_{t}.txt', 'w')

    def _copy_code_path(self, source_path: str, code_dir: str):
        source_path = os.path.normpath(source_path)
        if not os.path.exists(source_path):
            return
        if os.path.isfile(source_path):
            if not (source_path.endswith('.py') or source_path.endswith('.yaml')):
                return
            dst_path = os.path.join(code_dir, source_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(source_path, dst_path)
            return
        for root, dirs, files in os.walk(source_path):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            rel_root = os.path.normpath(root)
            dst_root = os.path.join(code_dir, rel_root)
            os.makedirs(dst_root, exist_ok=True)
            for file_name in files:
                if not (file_name.endswith('.py') or file_name.endswith('.yaml')):
                    continue
                src_file = os.path.join(root, file_name)
                dst_file = os.path.join(dst_root, file_name)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)

    def save_codes(self):
        if len(hparams['save_codes']) > 0:
            t = datetime.now().strftime('%Y%m%d%H%M%S')
            code_dir = os.path.join(self.work_dir, 'codes', t)
            os.makedirs(code_dir, exist_ok=True)
            for c in hparams['save_codes']:
                self._copy_code_path(c, code_dir)
            print(f"| Copied codes to {code_dir}.")
