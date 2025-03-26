import json
import copy
import functools
import os
from PIL import Image
import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .BinaryMNISTClassifier import BinaryMNISTClassifier
from .MNISTClassifier import UnifiedRGBMNISTClassifier
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
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
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = 500
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self.mnist_classifier = UnifiedRGBMNISTClassifier()
        self.mnist_classifier.model.eval()  # Set to evaluation mode

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
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

    def compute_rewards(self, img, metadata_path=None):
        assert metadata_path is not None, "metadata_path is required"
        #FIXME: assert batch size is 1
        assert self.batch_size == 1, "batch size should be 1"
        char_images, characters = self.decompose_image(img, metadata_path)
        
        is_correct, mnist_digit_count, confidences, masked_number_length = self.check_equation_correctness(char_images, characters)
        # Compute reward based on MNIST digit presence and equation correctness
        mnist_digit_reward = mnist_digit_count / max(len(characters) - 2, 1)  # Fraction of valid MNIST digits
        correctness_reward = 1.0 if is_correct else -1.0
        confidence_reward = sum(confidences) / max(len(characters) - 2, 1) if confidences else 0
        
        logger.log(f"Step: {self.step}, MNIST Digit Reward: {mnist_digit_reward}, Correctness Reward: {correctness_reward}, Confidence Reward: {confidence_reward}")
        # total_reward = 0.5 * mnist_digit_reward + 0.3 * correctness_reward + 0.2 * confidence_reward
        total_reward = mnist_digit_reward
        return th.tensor([total_reward], device=dist_util.dev())

    def check_equation_correctness(self, char_images, characters):
        equation = ""
        mnist_digit_count = 0
        confidences = []
        entropy_penalty = 0.0
        
        # create a new folder to save predicted images exists is ok
        os.makedirs('predicted_images', exist_ok=True)
        for char_img, char in zip(char_images, characters):
            if char in ["+", "*", "="]:  
                equation += char
            else:
                # Convert to RGB and resize to 28x28 (as required by classifier)
                char_img_pil = char_img.convert("RGB").resize((28, 28))

                # Save for debugging
                char_img_pil.save(os.path.join('predicted_images', f"{char}.png"))

                # Predict digit using the classifier
                with th.no_grad():
                    predicted_label, confidence = self.mnist_classifier.predict(char_img_pil)
                logger.info(f"Predicted Label: {predicted_label}, Confidence: {confidence}")

                if confidence > 0.5 and predicted_label != 10:
                    mnist_digit_count += 1
                    equation += str(predicted_label)
                else:
                    equation += '-1'  # Placeholder for uncertain digits
                    # save to a folder contains failed images with timestamp as its name
                # char_img_pil.save(f'/users/zzhan513/data/zzhan513/visual_reasoning/train_repaint/guided-diffusion/char_img.png')


                confidences.append(confidence)
                # entropy_penalty += self.compute_entropy(predicted_probs)  # Compute entropy

        try:
            left_side, right_side = equation.split("=")
            is_correct = eval(left_side) == eval(right_side)
        except:
            is_correct = False
        # log probs, label and entropy penalty
        # LOGGER.info(f"Confidences: {confidences}")
        # LOGGER.info(f"Entropy Penalty: {entropy_penalty}")
        # LOGGER.info(f"Equation: {equation}")
        return is_correct, mnist_digit_count, confidences, len(confidences)
    
    def decompose_image(self, image, metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        char_images = []
        characters = []

        # extract each character using the bounding boxes
        for bbox in metadata["bboxes"]:
            character = bbox["character"]
            left = bbox["left"]
            top = bbox["top"]
            right = bbox["right"]
            bottom = bbox["bottom"]
            char_image = image.crop((left, top, right, bottom))
            # save char image
            # if not os.path.exists('char_images'):
            #     os.makedirs('char_images')
            # print(f"Step: {self.step}, BBox: {left, top, right, bottom}, Character: {character}, char_image{char_image}")
            # char_image.save(os.path.join('char_images', f"char_{character}_{left}.png"))
            char_images.append(char_image)
            characters.append(character)            
        # image.save(os.path.join('char_images', f"original.png"))
        # exit()
        return char_images, characters

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        assert batch.shape[0] == 1, "batch size should be 1"

        # FIXME: hard-coded path for metadata
        metadata_path = '/users/zzhan513/data/zzhan513/visual_reasoning/train_repaint/guided-diffusion/mnist_addition_input/metadata'
        # getting ith path from metadata
        metadata_files = [f for f in os.listdir(metadata_path) if os.path.isfile(os.path.join(metadata_path, f))]
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            # Compute rewards based on generated inpainted digits
            generated_images = self.diffusion.p_sample_loop(
                self.model,
                (micro.shape[0], 3, 256, 256),
                clip_denoised=True,
                model_kwargs=micro_cond,
            )
            # FIXME: assert batch size is 1
            assert generated_images.shape[0] == 1
            img = generated_images[0].unsqueeze(0)  # Take first sample
            img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8) 
            img = img.permute(0, 2, 3, 1)
            img = img.contiguous()
            img = img.cpu().numpy()
            img = img.squeeze(0)
            img = Image.fromarray(img)
            if self.step % 5 == 0:
                if not os.path.exists('convert_generated_to_greyscale_image_checking'):
                    os.makedirs('convert_generated_to_greyscale_image_checking')
                img.save(os.path.join('convert_generated_to_greyscale_image_checking', f"sample_step_{self.step}_idx{i}_original.png"))
            img = img.convert('L')
            # save generated images every 500 steps
            if self.step % 5 == 0:
                if not os.path.exists('convert_generated_to_greyscale_image_checking'):
                    os.makedirs('convert_generated_to_greyscale_image_checking')
                img.save(os.path.join('convert_generated_to_greyscale_image_checking', f"sample_step_{self.step}_idx{i}_greyscale.png"))
            json_path = os.path.join(metadata_path, metadata_files[i])
            rewards = self.compute_rewards(img, json_path)
            logger.log(f"Step: {self.step}, Reward: {rewards.item()}")
            
            logger.log(f'original losses: {(losses["loss"] * weights).mean()}')
            loss = (losses["loss"] * weights).mean() * th.exp(-rewards.mean()) 
            logger.log(f'scaled losses: {loss.item()}')
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
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
