# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import math
import os
from contextlib import nullcontext
from functools import partial
from typing import Optional
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
# import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state

from cosmos_predict1.diffusion.training.callbacks.every_n import EveryN
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.easy_io import easy_io
from cosmos_predict1.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_predict1.utils.visualize.video import save_img_or_video
from cosmos_predict1.diffusion.training.context_parallel import split_inputs_cp
# from cosmos_predict1.diffusion.training.datasets.data_sources.item_datasets_for_validation import get_itemdataset_option
from cosmos_predict1.diffusion.training.utils.fsdp_helper import possible_fsdp_scope

# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        is_x0: bool = True,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        show_all_frames: bool = False,
        text_embedding_type: str = "t5_xxl",
    ):
        super().__init__(every_n, step_size)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_x0 = is_x0
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.text_embedding_type = text_embedding_type
        self.show_all_frames = show_all_frames
        self.rank = distributed.get_rank()

    def on_train_start(self, model: torch.nn.Module, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if self.fix_batch is not None:
            with misc.timer(f"loading fix_batch {self.fix_batch}"):
                self.fix_batch = misc.co(easy_io.load(self.fix_batch), "cpu")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            negative_prompt_path = "negative_prompt_data.pkl"
            if os.path.exists(negative_prompt_path):
                self.negative_prompt_data = easy_io.load(negative_prompt_path)
            else:
                raise FileNotFoundError(f"Negative prompt data not found at {negative_prompt_path}")

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)
        # TODO: (qsh 2024-07-01) this may be problematic due to sometimes we have uncondition, some times we have condition due to cfg dropout
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        if not model.is_image_batch(data_batch):
            if parallel_state.get_context_parallel_world_size() > 1:
                raw_data = split_inputs_cp(raw_data, seq_dim=2, cp_group=model.net.cp_group)
                x0 = split_inputs_cp(x0, seq_dim=2, cp_group=model.net.cp_group)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_x0_level + 1)[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )
        # check if it has cotrol input
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            to_show.append(hint.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        # Dump model state_dict to debug_{rank}.pt
        torch.save(model.state_dict(), f"every_n_impl_debug_iter{iteration}_rank{self.rank}.pt")
        log.info(f"Model state_dict saved to every_n_impl_debug_iter{iteration}_rank{self.rank}.pt")
        log.info(f"EveryNDrawSample: every_n_impl, is_ema: {self.is_ema}")
        if self.is_ema:
            log.info(f"model ema enabled: {model.config.ema.enabled}")
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                output_path = f"{self.local_dir}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json"
                log.info(f"Saving batch info to {output_path}")
                with open(output_path, "w") as f:
                    json.dump(batch_info, f, indent=4)

        log.info("entering, every_n_impl", rank0_only=False)
        with context():
            log.info("entering, ema", rank0_only=False)
            with possible_fsdp_scope(model.model):
                # we only use rank0 and rank to generate images and save
                # other rank run forward pass to make sure it works for FSDP
                log.info("entering, fsdp", rank0_only=False)
                if self.is_x0:
                    log.info("entering, x0_pred", rank0_only=False)
                    x0_img_fp, mse_loss, sigmas = self.x0_pred(
                        trainer,
                        model,
                        data_batch,
                        output_batch,
                        loss,
                        iteration,
                    )
                    log.info("done, x0_pred", rank0_only=False)
                    if self.save_s3 and self.rank == 0:
                        easy_io.dump(
                            {
                                "mse_loss": mse_loss.tolist(),
                                "sigmas": sigmas.tolist(),
                                "iteration": iteration,
                            },
                            f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                        )
                if self.is_sample:
                    log.info("entering, sample", rank0_only=False)
                    sample_img_fp = self.sample(
                        trainer,
                        model,
                        data_batch,
                        output_batch,
                        loss,
                        iteration,
                    )
                    log.info("done, sample", rank0_only=False)
                if self.fix_batch is not None:
                    misc.to(self.fix_batch, "cpu")

                log.info("waiting for all ranks to finish", rank0_only=False)
                dist.barrier()
        # if wandb.run:
        #     sample_counter = getattr(trainer, "sample_counter", iteration)
        #     data_type = "image" if model.is_image_batch(data_batch) else "video"
        #     tag += f"_{data_type}"
        #     info = {
        #         "trainer/global_step": iteration,
        #         "sample_counter": sample_counter,
        #     }
        #     if self.is_x0:
        #         info[f"{self.name}/{tag}_x0"] = wandb.Image(x0_img_fp, caption=f"{sample_counter}")
        #         # convert mse_loss to a dict
        #         mse_loss = mse_loss.tolist()
        #         info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

        #     if self.is_sample:
        #         info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp, caption=f"{sample_counter}")
        #     wandb.log(
        #         info,
        #         step=iteration,
        #     )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            data_batch = misc.to(self.fix_batch, **model.tensor_kwargs)

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        if self.use_negative_prompt:
            if self.text_embedding_type == "t5_xxl":
                negative_prompt_key = "t5_text_embeddings"
            else:
                negative_prompt_key = f"{self.text_embedding_type}_text_embeddings"

            batch_size = x0.shape[0]
            data_batch["neg_t5_text_embeddings"] = misc.to(
                repeat(
                    self.negative_prompt_data[negative_prompt_key],
                    "... -> b ...",
                    b=batch_size,
                ),
                **model.tensor_kwargs,
            )
            assert (
                data_batch["neg_t5_text_embeddings"].shape == data_batch["t5_text_embeddings"].shape
            ), f"{data_batch['neg_t5_text_embeddings'].shape} != {data_batch['t5_text_embeddings'].shape}"
            data_batch["neg_t5_text_mask"] = data_batch["t5_text_mask"]
            # neg_t5_text_mask = torch.zeros_like(data_batch["t5_text_mask"]) # [B, 512]
            # neg_t5_text_mask[:, :132] = 1
            # data_batch["neg_t5_text_mask"] = neg_t5_text_mask

        to_show = []
        # for guidance in [3.0, 7.0, 9.0, 13.0]:
        for guidance in [7.0]:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                is_negative_prompt=True if self.use_negative_prompt else False,
            )
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())

        to_show.append(raw_data.float().cpu())
        # visualize input video
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            for idx in range(0, hint.size(1), 3):
                x_rgb = hint[:, idx : idx + 3]
                to_show.append(x_rgb.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = output_batch["x0"].shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path
        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            output_path = f"{self.local_dir}/{base_fp_wo_ext}"
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                output_path,
            )

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        # if self.rank == 0 and wandb.run:
        #     if is_single_frame:  # image case
        #         to_show = rearrange(
        #             to_show[:, :n_viz_sample],
        #             "n b c t h w -> t c (n h) (b w)",
        #         )
        #         image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
        #         # resize so that wandb can handle it
        #         torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
        #     else:
        #         to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]
        #         if not self.show_all_frames:
        #             # resize 3 frames frames so that we can display them on wandb
        #             _T = to_show.shape[3]
        #             three_frames_list = [0, _T // 2, _T - 1]
        #             to_show = to_show[:, :, :, three_frames_list]
        #             log_image_size = 1024
        #         else:
        #             log_image_size = 512 * to_show.shape[3]
        #         to_show = rearrange(
        #             to_show,
        #             "n b c t h w -> 1 c (n h) (b t w)",
        #         )

        #         # resize so that wandb can handle it
        #         image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
        #         torchvision.utils.save_image(
        #             resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True
        #         )

        #     return local_path
        return None
