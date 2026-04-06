from contextlib import nullcontext
import copy
from numbers import Number

import numpy as np

try:
    from torch.amp import autocast
    _TORCH_AMP_NEW_API = True
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast
    _TORCH_AMP_NEW_API = False

import torch
import torch.distributed as dist
import tqdm

from utils.commons.hparams import hparams
from utils.commons.tensor_utils import move_to_cuda


class TrainerLoopMixin:
    """Training/eval loop extracted from Trainer for easier inspection."""

    def _amp_enabled(self) -> bool:
        return bool(self.amp and self.on_gpu and torch.cuda.is_available())

    def _build_autocast_kwargs(self) -> dict:
        kwargs = {"enabled": self._amp_enabled()}
        if _TORCH_AMP_NEW_API:
            kwargs["device_type"] = 'cuda'
        return kwargs

    def _prepare_batch(self, batch):
        if not self.on_gpu:
            return batch
        return move_to_cuda(copy.copy(batch), self.root_gpu)

    @staticmethod
    def _set_dataloader_epoch(dataloader, epoch: int):
        for sampler in (
            getattr(dataloader, 'batch_sampler', None),
            getattr(dataloader, 'sampler', None),
        ):
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
                return

    def _set_trainable_params_for_optimizer(self, task_ref, optimizer):
        if len(self.optimizers) <= 1:
            return
        for param in task_ref.parameters():
            param.requires_grad = False
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = True

    def _should_step_optimizer(self) -> bool:
        return (self.global_step + 1) % self.accumulate_grad_batches == 0

    def _ddp_sync_context(self, *, should_step: bool):
        if self.use_ddp and hasattr(self.task, 'no_sync') and not should_step:
            return self.task.no_sync()
        return nullcontext()

    def _check_nan_grads(self, task_ref):
        if not self.print_nan_grads:
            return
        has_nan_grad = False
        for name, param in task_ref.named_parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                print("| NaN params: ", name, param, param.grad)
                has_nan_grad = True
        if has_nan_grad:
            raise RuntimeError("NaN gradients detected.")

    @staticmethod
    def _collapse_metrics(metric_dicts):
        return {k: v for d in metric_dicts for k, v in d.items()}

    @staticmethod
    def _accumulate_weighted_scalars(weighted_totals, source_dict, weight):
        if not isinstance(source_dict, dict):
            return
        for key, value in source_dict.items():
            if isinstance(value, Number) or np.isscalar(value):
                weighted_totals[key] = weighted_totals.get(key, 0.0) + float(value) * float(weight)

    def _aggregate_eval_results_across_ranks(self, eval_results):
        if not (self.use_ddp and dist.is_available() and dist.is_initialized()):
            return eval_results
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, eval_results)
        non_empty_results = [result for result in gathered_results if result is not None]
        if len(non_empty_results) == 0:
            return eval_results

        total_weight = 0.0
        aggregated = {}
        tb_totals = {}
        scalar_totals = {}
        for result in non_empty_results:
            raw_weight = result.get('__eval_nsamples', None) if isinstance(result, dict) else None
            if raw_weight is None:
                weight = 1.0
            else:
                weight = float(raw_weight)
                if weight <= 0.0:
                    continue
            total_weight += weight
            if isinstance(result, dict):
                self._accumulate_weighted_scalars(tb_totals, result.get('tb_log', {}), weight)
                scalar_fields = {
                    key: value for key, value in result.items()
                    if key != 'tb_log' and not str(key).startswith('__')
                }
                self._accumulate_weighted_scalars(scalar_totals, scalar_fields, weight)
                for key, value in result.items():
                    if key == 'tb_log' or str(key).startswith('__'):
                        continue
                    if key in aggregated:
                        continue
                    if not (isinstance(value, Number) or np.isscalar(value)):
                        aggregated[key] = value

        if total_weight <= 0.0:
            return non_empty_results[0]
        if tb_totals:
            aggregated['tb_log'] = {key: value / total_weight for key, value in tb_totals.items()}
        for key, value in scalar_totals.items():
            aggregated[key] = value / total_weight
        aggregated['__eval_nsamples'] = int(round(total_weight))
        return aggregated

    def run_evaluation(self, test=False):
        eval_results = self.evaluate(
            self.task,
            test,
            tqdm_desc='Valid' if not test else 'test',
            max_batches=hparams['eval_max_batches'],
        )
        eval_results = self._aggregate_eval_results_across_ranks(eval_results)
        if eval_results is not None and 'tb_log' in eval_results:
            self.log_metrics_to_tb(eval_results['tb_log'])
        if self.proc_rank == 0 and not test:
            self.save_checkpoint(epoch=self.current_epoch, logs=eval_results)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        if max_batches == -1:
            max_batches = None
        task.zero_grad(set_to_none=True)
        task.eval()
        torch.set_grad_enabled(False)

        task_ref = self.get_task_ref()
        if test:
            ret = task_ref.test_start()
            if ret == 'EXIT':
                return None
        else:
            task_ref.validation_start()
        outputs = []
        dataloader = task_ref.test_dataloader() if test else task_ref.val_dataloader()
        pbar = tqdm.tqdm(
            dataloader,
            desc=tqdm_desc,
            total=max_batches,
            dynamic_ncols=True,
            unit='step',
            disable=self.root_gpu > 0,
        )
        for batch_idx, batch in enumerate(pbar):
            if batch is None:  # pragma: no cover
                continue
            if max_batches is not None and batch_idx >= max_batches:
                break
            prepared_batch = self._prepare_batch(batch)
            args = [prepared_batch, batch_idx]
            if self.use_ddp:
                output = task(*args)
            else:
                output = task_ref.test_step(*args) if test else task_ref.validation_step(*args)
            outputs.append(output)
        eval_results = task_ref.test_end(outputs) if test else task_ref.validation_end(outputs)
        task.train()
        torch.set_grad_enabled(True)
        return eval_results

    def train(self):
        task_ref = self.get_task_ref()
        task_ref.on_train_start()
        if self.num_sanity_val_steps > 0:
            self.evaluate(self.task, False, 'Sanity Val', max_batches=self.num_sanity_val_steps)
        if self.on_gpu:
            torch.cuda.empty_cache()
        dataloader = task_ref.train_dataloader()
        epoch = self.current_epoch
        while True:
            self._set_dataloader_epoch(dataloader, epoch)
            task_ref.current_epoch = epoch
            self.current_epoch = epoch
            self.batch_loss_value = 0
            task_ref.on_epoch_start()

            train_pbar = tqdm.tqdm(
                dataloader,
                initial=self.global_step,
                total=float('inf'),
                dynamic_ncols=True,
                unit='step',
                disable=self.root_gpu > 0,
            )
            for batch_idx, batch in enumerate(train_pbar):
                if self.global_step % self.val_check_interval == 0 and not self.fisrt_epoch:
                    self.run_evaluation()
                pbar_metrics, tb_metrics = self.run_training_batch(batch_idx, batch)
                train_pbar.set_postfix(**pbar_metrics)
                self.fisrt_epoch = False
                if (self.global_step + 1) % self.tb_log_interval == 0:
                    self.log_metrics_to_tb(tb_metrics)

                self.global_step += 1
                task_ref.global_step = self.global_step
                if self.global_step >= self.max_updates:
                    print("| Training end..")
                    break
            task_ref.on_epoch_end()
            epoch += 1
            if self.global_step >= self.max_updates:
                break
        if (
            not self.testing
            and getattr(self, 'proc_rank', 0) == 0
            and self.global_step > 0
            and getattr(self, 'last_saved_ckpt_step', None) != self.global_step
        ):
            print(f"| Saving final checkpoint at step {self.global_step} ..")
            self.save_checkpoint(epoch=self.current_epoch, logs=None)
        task_ref.on_train_end()

    def run_training_batch(self, batch_idx, batch):
        if batch is None:
            return {}, {}
        all_progress_bar_metrics = []
        all_log_metrics = []
        task_ref = self.get_task_ref()
        amp_enabled = self._amp_enabled()
        autocast_kwargs = self._build_autocast_kwargs()
        should_step = self._should_step_optimizer()
        prepared_batch = self._prepare_batch(batch)

        for opt_idx, optimizer in enumerate(self.optimizers):
            if optimizer is None:
                continue
            self._set_trainable_params_for_optimizer(task_ref, optimizer)
            optimizer_batch = copy.copy(prepared_batch)
            sync_context = self._ddp_sync_context(should_step=should_step)
            with sync_context:
                with autocast(**autocast_kwargs):
                    args = [optimizer_batch, batch_idx, opt_idx]
                    if self.use_ddp:
                        output = self.task(*args)
                    else:
                        output = task_ref.training_step(*args)
                    if output is None:
                        continue
                    loss = output['loss']
                    if loss is None:
                        continue
                    progress_bar_metrics = output['progress_bar']
                    log_metrics = output['tb_log']
                    loss = loss / self.accumulate_grad_batches

                if loss.requires_grad:
                    if amp_enabled:
                        self.amp_scalar.scale(loss).backward()
                    else:
                        loss.backward()

            all_log_metrics.append(log_metrics)
            all_progress_bar_metrics.append(progress_bar_metrics)
            self._check_nan_grads(task_ref)

            if should_step:
                if amp_enabled:
                    self.amp_scalar.unscale_(optimizer)
                task_ref.on_before_optimization(opt_idx)
                if amp_enabled:
                    self.amp_scalar.step(optimizer)
                    self.amp_scalar.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)

        return self._collapse_metrics(all_progress_bar_metrics), self._collapse_metrics(all_log_metrics)
