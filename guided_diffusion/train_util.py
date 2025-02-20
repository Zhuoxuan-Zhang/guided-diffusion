import copy
import functools
import os
from PIL import Image
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


def compute_dpo_loss(model_losses_w, model_losses_l, ref_losses_w, ref_losses_l, beta=0.1):
    """
    Compute the Direct Preference Optimization (DPO) loss using both the model and reference model.
    
    Args:
        model_losses_w (Tensor): Preferred losses from the model.
        model_losses_l (Tensor): Rejected losses from the model.
        ref_losses_w (Tensor): Preferred losses from the reference model.
        ref_losses_l (Tensor): Rejected losses from the reference model.
        beta (float): Scaling term.

    Returns:
        Tensor: The computed DPO loss.
    """
    model_diff = model_losses_w - model_losses_l
    ref_diff = ref_losses_w - ref_losses_l

    scale_term = -0.5 * beta
    inside_term = scale_term * (model_diff - ref_diff)

    # Clamp values for numerical stability before applying logsigmoid
    inside_term = th.clamp(inside_term, min=-10, max=10)

    loss = -th.nn.functional.logsigmoid(inside_term).mean()

    return loss


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        reference_model,
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
        self.reference_model = reference_model
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
        self.loss_pos = None
        self.loss_neg = None
        self.dpo_loss = None
        self.overall_loss = None

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
            self.mp_trainer.master_params,
            lr=self.lr * 0.5,
            weight_decay=self.weight_decay,
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
            self.ddp_reference_model = DDP(
                self.reference_model,
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
            self.ddp_reference_model = self.reference_model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log(f"device: {dist_util.dev()}")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
                self.reference_model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.reference_model.parameters())

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
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch_pos, batch_neg, cond_pos, cond_neg):
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
            t_neg, weights_neg = self.schedule_sampler.sample(
                micro_neg.shape[0], dist_util.dev()
            )

            # compute positive and negative losses
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_pos,
                t_pos,
                model_kwargs=micro_pos_cond,
            )
            if last_batch or not self.use_ddp:
                losses_pos = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses_pos = compute_losses()

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_neg,
                t_neg,
                model_kwargs=micro_neg_cond,
            )
            if last_batch or not self.use_ddp:
                losses_neg = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses_neg = compute_losses()

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_reference_model,
                micro_pos,
                t_pos,
                model_kwargs=micro_pos_cond,
            )
            if last_batch or not self.use_ddp:
                ref_losses_pos = compute_losses()
            else:
                with self.ddp_reference_model.no_sync():
                    ref_losses_pos = compute_losses()

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_reference_model,
                micro_neg,
                t_neg,
                model_kwargs=micro_neg_cond,
            )
            if last_batch or not self.use_ddp:
                ref_losses_neg = compute_losses()
            else:
                with self.ddp_reference_model.no_sync():
                    ref_losses_neg = compute_losses()
            
            model_losses_w, model_losses_l = losses_pos["loss"], losses_neg["loss"]
            ref_losses_w, ref_losses_l = ref_losses_pos["loss"], ref_losses_neg["loss"]
            # compute DPO loss
            dpo_loss = compute_dpo_loss(model_losses_w, model_losses_l, ref_losses_w, ref_losses_l, beta=0.08)

            # overall_loss = loss_pos + dpo_loss

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t_pos, dpo_loss.detach()
                )
            # log the loss
            self.loss_pos = model_losses_w
            self.loss_neg = model_losses_l
            self.dpo_loss = dpo_loss

            # save an example image every 1000 steps
            if self.step % 1000 == 0:
                generated_images = self.diffusion.p_sample_loop(
                    self.model,
                    (micro_pos.shape[0], 3, 256, 256),
                    clip_denoised=True,
                    model_kwargs=micro_pos_cond,
                    )
                img = generated_images[0].unsqueeze(0)  # Take first sample
                img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8) 
                img = img.permute(0, 2, 3, 1)
                img = img.contiguous()
                img = img.cpu().numpy()
                img = img.squeeze(0)
                img = Image.fromarray(img)
                if not os.path.exists('image_checking'):
                    os.makedirs('image_checking')
                img.save(os.path.join('image_checking', f"sample_step_{self.step}_idx{i}.png"))

            self.mp_trainer.backward(dpo_loss)

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
        logger.logkv("preferred_loss", self.loss_pos.item())
        logger.logkv("rejected_loss", self.loss_neg.item())
        logger.logkv("dpo_loss", self.dpo_loss.item())
        # logger.logkv("overall_loss", self.overall_loss.item())

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
