from modules.Conan.Conan import Conan, ConanPostnet
from modules.tts.iclspeech.multi_window_disc import Discriminator
from tasks.Conan.base_gen_task import AuxDecoderMIDITask
from tasks.Conan.dataset import ConanDataset
from tasks.Conan.rhythm.task_mixin import RhythmConanTaskMixin
from modules.Conan.rhythm.stages import detect_rhythm_stage
from utils.commons.hparams import hparams
import torch
import torch.nn as nn


class ConanEmbTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset

    def build_tts_model(self):
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def run_model(self, sample):
        with torch.no_grad():
            ref = sample['mels']
            output = self.model.encode_spk_embed(ref.transpose(1, 2)).squeeze(2)
        return {}, {"style_embed": output}


class ConanTask(RhythmConanTaskMixin, AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        self._warned_retimed_pitch_supervision = False
        self._disc_skip_for_retimed = False
        self.mel_disc = None
        self.disc_params = []
        self._validate_rhythm_training_hparams()
        self.build_disc_model()

    def build_tts_model(self):
        self.model = Conan(0, hparams)
        stage = detect_rhythm_stage(hparams)
        teacher_only_stage = stage == "teacher_offline" and not getattr(self.model, "rhythm_enable_v3", False)
        if teacher_only_stage:
            teacher_params = self._collect_offline_teacher_gen_params()
            self.gen_params = self._freeze_to_trainable_params(
                teacher_params,
                stage_name="teacher_offline",
            )
        elif bool(hparams.get("rhythm_optimize_module_only", False)):
            rhythm_params = self._collect_rhythm_gen_params()
            self.gen_params = self._freeze_to_trainable_params(
                rhythm_params,
                stage_name=stage,
            )
        else:
            self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def _freeze_to_trainable_params(self, params, *, stage_name: str):
        if len(params) <= 0:
            raise RuntimeError(
                f"Stage '{stage_name}' resolved zero trainable params. "
                "Refusing to silently fall back to full-model optimization because that would break "
                "stage isolation and invalidate gradient-scope checks."
            )
        trainable_param_ids = {id(param) for param in params}
        frozen_params = []
        for param in self.model.parameters():
            selected = id(param) in trainable_param_ids
            param.requires_grad = selected
            if selected:
                frozen_params.append(param)
        if len(frozen_params) <= 0:
            raise RuntimeError(
                f"Stage '{stage_name}' lost all trainable params after freeze application. "
                "Check rhythm/offline-teacher parameter collection against the current model structure."
            )
        if len({id(param) for param in frozen_params}) != len(trainable_param_ids):
            raise RuntimeError(
                f"Stage '{stage_name}' collected params that do not belong to the active model instance. "
                "Refusing to continue with a partially matched parameter set."
            )
        return frozen_params

    def build_disc_model(self):
        disc_win_num = int(hparams.get('disc_win_num', 0) or 0)
        lambda_mel_adv = float(hparams.get('lambda_mel_adv', 0.0) or 0.0)
        if lambda_mel_adv <= 0.0 or disc_win_num <= 0:
            self.mel_disc = None
            self.disc_params = []
            return
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        disc_scheduler = None
        if optimizer[1] is not None:
            disc_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[1],
                **hparams["discriminator_scheduler_params"],
            )
        return [
            super().build_scheduler(optimizer[0]),
            disc_scheduler,
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        elif len(self.disc_params) > 0:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, (list, tuple)):
            if 0 <= optimizer_idx < len(self.scheduler) and self.scheduler[optimizer_idx] is not None:
                self.scheduler[optimizer_idx].step(self.global_step // hparams['accumulate_grad_batches'])
            return
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])


def self_clone(x):
    if x is None:
        return None
    y = x.clone()
    return torch.cat((x, y), dim=0)


class VCPostnetTask(ConanTask):
    def __init__(self):
        super(VCPostnetTask, self).__init__()
        self.drop_prob = hparams['drop_tech_prob']

    def build_model(self):
        self.build_pretrain_model()
        self.model = ConanPostnet()

    def build_pretrain_model(self):
        dict_size = 0
        self.pretrain = Conan(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True)
        for _, value in self.pretrain.named_parameters():
            value.requires_grad = False

    def run_model(self, sample, infer=False, noise=None, test=False):
        content = sample["content"]
        spk_embed = None
        f0, uv = sample["f0"], sample["uv"]
        target = sample["mels"]
        ref = sample['ref_mels']
        cfg = False
        output = self.pretrain(content, spk_embed=spk_embed, target=target, ref=ref, f0=f0, uv=uv, infer=infer)

        self.model(target, infer, output, cfg, cfg_scale=hparams['cfg_scale'], noise=noise)
        losses = {}
        losses["flow"] = output["flow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])
