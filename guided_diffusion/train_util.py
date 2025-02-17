import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def dpo_loss(preferred_loss, rejected_loss, beta=0.1):
    """Compute DPO loss with better numerical stability."""
    diff = (preferred_loss - rejected_loss) / (beta + 1e-8)
    diff = th.clamp(diff, min=-5, max=5)

    # log-sigmoid computation
    loss = -th.nn.functional.logsigmoid(diff).mean()

    # If NaN is detected, return zero loss
    if th.isnan(loss).any() or th.isinf(loss).any():
        print("NaN detected in DPO loss! Returning zero loss...")
        return loss.detach()  # Preserve gradient tracking
    return loss


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        preferred_data,
        reject_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.preferred_data = preferred_data
        self.reject_data = reject_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.last_loss_pos = None
        self.last_loss_neg = None
        self.last_dpo_loss = None

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr * 0.5, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch_pos, cond_pos = next(self.preferred_data)
            batch_neg, cond_neg = next(self.reject_data)

            self.run_step(batch_pos, batch_neg, cond_pos, cond_neg)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_pos, batch_neg, cond_pos, cond_neg):
        self.forward_backward(batch_pos, batch_neg, cond_pos, cond_neg)
        grad_norm = th.nn.utils.clip_grad_norm_(self.mp_trainer.master_params, max_norm=1.0)
        if grad_norm > 5.0:
            print(f"High gradient norm detected: {grad_norm}. Reducing learning rate...")
            for param_group in self.opt.param_groups:
                param_group["lr"] = max(param_group["lr"] * 0.9, 1e-6)   # Reduce learning rate by 10%
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch_pos, batch_neg, cond_pos, cond_neg):
        """Modified forward_backward to apply DPO loss."""
        self.mp_trainer.zero_grad()

        for i in range(0, batch_pos.shape[0], self.microbatch):
            micro_pos = batch_pos[i : i + self.microbatch].to(dist_util.dev())
            micro_neg = batch_neg[i : i + self.microbatch].to(dist_util.dev())
            micro_pos_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond_pos.items()
            }
            micro_neg_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond_neg.items()
            }
            last_batch = (i + self.microbatch) >= batch_pos.shape[0]
            t_pos, weights_pos = self.schedule_sampler.sample(
                micro_pos.shape[0], dist_util.dev()
            )
            t_neg, weights_neg = self.schedule_sampler.sample(micro_neg.shape[0], dist_util.dev())

            losses_pos = self.diffusion.training_losses(
                self.ddp_model, micro_pos, t_pos, model_kwargs=micro_pos_cond
            )
            losses_neg = self.diffusion.training_losses(
                self.ddp_model, micro_neg, t_neg, model_kwargs=micro_neg_cond
            )

            # check for NaNs before computing loss
            if th.isnan(losses_pos["loss"]).any() or th.isnan(losses_neg["loss"]).any():
                print("NaN detected in losses, skipping step...")
                self.mp_trainer.zero_grad()  # Reset gradients
                return
            beta = 0.02
            if self.step % 500 == 0:
                beta = max(beta * (0.99 ** self.step), 0.005)  # Gradually decrease beta

            valid_samples = losses_pos["loss"] < losses_neg["loss"]
            if not valid_samples.all():  # If any invalid samples exist
                # print(f"Warning: {(~valid_samples).sum().item()} invalid samples found. Filtering...")
                losses_pos["loss"] = losses_pos["loss"][valid_samples]
                losses_neg["loss"] = losses_neg["loss"][valid_samples]
                if losses_pos["loss"].numel() == 0:  # If no valid samples left, skip
                    self.mp_trainer.zero_grad()
                    return

            loss = dpo_loss(losses_pos["loss"], losses_neg["loss"], beta=beta)
            
            
            self.last_loss_pos = losses_pos["loss"].mean().detach()
            self.last_loss_neg = losses_neg["loss"].mean().detach()
            self.last_dpo_loss = loss.detach()

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("preferred_loss", self.last_loss_pos)
        logger.logkv("rejected_loss", self.last_loss_neg)
        logger.logkv("dpo_loss", self.last_dpo_loss.item())  # Log actual DPO loss

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
