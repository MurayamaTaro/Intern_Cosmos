# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks

from cosmos_predict1.autoregressive.callbacks.video_sampling_teacher_forcing import resize_image
from cosmos_predict1.autoregressive.configs.inference.inference_config import TrainingSamplingConfig as SamplingConfig
from cosmos_predict1.autoregressive.evaluation.utils import (
    create_color_edge,
    get_num_gen_tokens,
    try_compute_num_frames,
)
from cosmos_predict1.autoregressive.evaluation.video_generator import generate_parital_tokens_from_data_batch
from cosmos_predict1.autoregressive.utils.parallel import broadcast_data_batch_in_tp_cp_group
from cosmos_predict1.callbacks.every_n import EveryN
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.easy_io import easy_io
from cosmos_predict1.utils.model import Model
from cosmos_predict1.utils.trainer import Trainer


class VideoSamplingPartialTokens(EveryN):
    def __init__(
        self,
        every_n: int,
        task_condition: str,
        sampling_config: SamplingConfig,
        video_latent_shape: list = [8, 24, 40],
        latent_context_t_sizes: list[int] = [1],
        save_folder: Optional[str] = None,
        max_samples: int = 1,
        step_size: int = 1,
        num_file_to_log: int = 12,
        iteration_early_test: int = 10,
    ):
        r"""
        This callback enables us to perform teacher forcing inference on the training data.
        By teacher forcing, we mean providing ground truth video tokens as inputs, and simply asking the model
        to predict the next tokens. The predicted next tokens are then visualized. This does not perform
        autoregressive sampling.
        We also upload the downsampled video frames to wandb. Downsampling is needed for wandb to work fast.

        Args:
            every_n (int): Call this callback every_n steps
            sampling_config (SamplingConfig): Sampling configuration
            video_latent_shape (list): Shape of the video latent
            latent_context_t_sizes (list[int]): Number of condition tokens in T to use, total number of condition token will be t * H * W
            save_folder (str): Name of the local folder to save the video
            max_samples (int): Maximum number of samples to generate
            step_size (int): Number of steps taken for gradient accumulation. Global iteration number is
                iteration // self.step_size
            num_file_to_log (int): Number of files to log to wandb
            iteration_early_test (int): Number of iterations to run the callback for early testing purposes, we want to test the callback early in training to make sure it works, if it has issue we will know early instead of waiting for the callback to run at every_n steps
        """
        super().__init__(every_n, step_size)

        self.save_folder = save_folder if save_folder else self.__class__.__name__
        self.video_latent_shape = video_latent_shape
        self.sampling_config = sampling_config
        self.max_samples = max_samples
        self.latent_context_t_sizes = latent_context_t_sizes
        assert (
            max(latent_context_t_sizes) <= video_latent_shape[0]
        ), f"max(latent_context_t_sizes) {latent_context_t_sizes} should be less than video_latent_shape[0]"
        self.rank = distributed.get_rank()
        self.num_file_to_log = num_file_to_log
        self.iteration_early_test = iteration_early_test
        self.task_condition = task_condition

        log.info(
            f"Video Partial Token Callback: Initialized, with these parameters: video_latent_shape: {video_latent_shape}, latent_context_t_sizes: {latent_context_t_sizes}, save_folder: {self.save_folder}, max_samples: {max_samples}, step_size: {step_size}, num_file_to_log: {num_file_to_log}, iteration_early_test: {iteration_early_test}, task_condition: {task_condition}"
        )

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.save_folder}"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Video Partial Token Callback: local_dir: {self.local_dir}")

    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        trainer = self.trainer
        global_step = iteration // self.step_size
        should_run = (
            global_step % self.every_n == 0 or iteration == self.iteration_early_test
        )  # Run the callback at every_n steps or at iteration_early_test (this is for early testing incase the callback fails)
        if should_run:
            log.debug(f"Callback {self.__class__.__name__} fired on train_batch_end step {global_step}")
            self.every_n_impl(trainer, model, data_batch, output_batch, loss, iteration)
            log.debug(f"Callback {self.__class__.__name__} finished on train_batch_end step {global_step}")
            # add necessary barrier to avoid timeout
            if self.barrier_after_run:
                distributed.barrier()

    @torch.inference_mode()
    def every_n_impl(
        self,
        trainer: Trainer,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        broadcast_data_batch_in_tp_cp_group(data_batch)

        dataset_name = data_batch.get("dataset_name", None)
        if dataset_name is not None and dataset_name.startswith("image"):
            # we disable the callback if the input video is an image batch
            log.info(f"dataset_name is {dataset_name}, skip this callback")
            return

        # get the caption
        captions = data_batch.get("caption", None)
        log.info(f"Video Partial Token Callback: iteration: {iteration}, captions: {captions}")

        # get the action
        action = data_batch.get("action", None)
        if self.task_condition == "video_and_action":
            log.info(f"Video Partial Token Callback: iteration: {iteration}, action: {action}")

        tokenizer_config = self.config.model.tokenizer_config
        for latent_context_t_size in self.latent_context_t_sizes:
            latent_chunk_duration = tokenizer_config.video_tokenizer.config.latent_chunk_duration
            pixel_chunk_duration = tokenizer_config.video_tokenizer.config.pixel_chunk_duration

            num_gen_tokens = get_num_gen_tokens(self.task_condition, latent_context_t_size, self.video_latent_shape)
            num_gen_frames, num_condition_frame = try_compute_num_frames(
                num_gen_tokens,
                latent_chunk_duration,
                pixel_chunk_duration,
                self.video_latent_shape,
                num_chunks_to_generate=1,
            )

            out_videos_LCHW = []
            with misc.timer(
                f"generate_parital_tokens_from_data_batch, num_frame={num_gen_frames}, num_gen_tokens={num_gen_tokens}"
            ):
                log.info(
                    f"self.sampling_config: {self.sampling_config}, tokenizer_config: {tokenizer_config}, self.video_latent_shape: {self.video_latent_shape}, max_samples: {self.max_samples}, latent_context_t_size: {latent_context_t_size}, task_condition: {self.task_condition}"
                )
                out_videos_cur_batch, inp_videos_cur_batch, _, _ = generate_parital_tokens_from_data_batch(
                    model,
                    data_batch,
                    num_gen_tokens,
                    self.sampling_config,
                    tokenizer_config,
                    self.video_latent_shape,
                    task_condition=self.task_condition,
                    max_samples=self.max_samples,
                    latent_context_t_size=latent_context_t_size,
                    action=action,
                )  # Each entry is [CLHW], range [0, 1]

            if inp_videos_cur_batch is None:
                log.warning("No input videos found, use all-zeros for the inp_videos_cur_batch")
                inp_videos_cur_batch = [torch.zeros_like(out_video) for out_video in out_videos_cur_batch]

            for out_video, input_video in zip(out_videos_cur_batch, inp_videos_cur_batch):
                _, L, _, W = out_video.shape
                color_edge = create_color_edge(
                    L, W, out_video.device, num_gen_frames=num_gen_frames, pixel_chunk_duration=pixel_chunk_duration
                )
                compose_video = torch.cat([color_edge, out_video, color_edge, input_video], dim=2)
                compose_video_LCHW = rearrange(compose_video, "C L H W -> L C H W")
                compose_video_LCHW_resize = resize_image(compose_video_LCHW, resize_factor=0.5)
                out_videos_LCHW.append(compose_video_LCHW_resize)  # [L, C, H, W], [0, 1]

            out_video_LCHW = torch.cat(out_videos_LCHW, dim=3)  # [L, C, H, W], [0, 1]
            # Save the whole video
            vid_save_path = f"{self.local_dir}/vid_partial_tokens_condframe_{num_condition_frame}_tokengen_{num_gen_tokens}_iter_{iteration:09d}_rank_{self.rank}.mp4"

            # Save files locally
            skip_save_file = False
            scaled_num_rank_to_log = (
                self.num_file_to_log
                * parallel_state.get_context_parallel_world_size()
                * parallel_state.get_tensor_model_parallel_world_size()
            )
            if parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()
                if self.rank != min(get_process_group_ranks(cp_group)):
                    skip_save_file = True
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # Turn on TP
                tp_group = parallel_state.get_tensor_model_parallel_group()
                if self.rank != min(get_process_group_ranks(tp_group)):
                    skip_save_file = True
            if self.rank < scaled_num_rank_to_log and not skip_save_file:
                easy_io.dump(
                    (rearrange(out_video_LCHW, "L C H W -> L H W C") * 255).to(torch.uint8).cpu().numpy(),
                    vid_save_path,
                    format="mp4",
                    fps=data_batch["fps"][0].item() if "fps" in data_batch else 24,
                )
                if captions is not None:
                    current_caption = " \n".join(captions[: len(out_videos_cur_batch)])
                    caption_save_path = vid_save_path.replace(".mp4", ".json")
                    easy_io.dump({"real_videos_name": vid_save_path, "caption": current_caption}, caption_save_path)

                log.info(f"Video Partial Token Callback: saved video to {vid_save_path}")
            dist.barrier()
            # Log to wandb
            if self.rank == 0 and wandb.run:
                files = glob.glob(
                    f"{self.local_dir}/vid_partial_tokens_condframe_{num_condition_frame}_tokengen_{num_gen_tokens}_iter_{iteration:09d}_rank_*.mp4"
                )
                files = sorted(files)[: self.num_file_to_log]
                log.info(f"glob number of files: {len(files)}, e.g. {files[0]}")
                # Open all the mp4, concat them and save as a single video
                if captions is None:
                    captions = [f"vid_partial_tokens_condframe_{num_condition_frame}_tokengen_{num_gen_tokens}"] * len(
                        files
                    )
                else:
                    captions = [
                        json.load(open(vid_save_path_i.replace(".mp4", ".json"), "r"))["caption"]
                        for vid_save_path_i in files
                    ]
                log_list = []
                for vid_save_path, caption in zip(files, captions):
                    log.info(f"vid: {vid_save_path} caption: {caption}")

                    log_list.append(wandb.Video(vid_save_path, caption=caption))
                wandb.log(
                    {"vid/partial_tokens": log_list},
                    step=iteration,
                )
