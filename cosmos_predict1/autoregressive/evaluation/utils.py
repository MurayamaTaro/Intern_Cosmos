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

import importlib
import os
import re
from pathlib import Path
from typing import Optional

import decord
import numpy as np
import torch
import torch.distributed as dist
from decord import VideoReader

from cosmos_predict1.utils import log

_COLOR_EDGE_WIDTH = 6


def create_color_edge(L, W, device, num_gen_frames=None, pixel_chunk_duration=None):
    """Create color edge to indicate the generated frames id on top of the video.
    - Red for input frames
    - Green for generated frames
    - Yellow for first frame of each chunk, i.e., i frame
    Args:
        L (int): Number of frames
        W (int): Width of the frame
        device (torch.device): Device
        num_gen_frames (int): Number of generated frames, if None, all frames are white
        pixel_chunk_duration (int): Duration of each chunk,
    Returns:
        color_edge (torch.Tensor): Color edge, shape [3, L, _COLOR_EDGE_WIDTH, W]
    """
    # Log input parameters
    log.info("Creating color edge with parameters:")
    log.info(f"  Number of frames (L): {L}")
    log.info(f"  Frame width (W): {W}")
    log.info(f"  Device: {device}")
    log.info(f"  Number of generated frames: {num_gen_frames}")
    log.info(f"  Pixel chunk duration: {pixel_chunk_duration}")

    color_edge = torch.zeros(3, L, _COLOR_EDGE_WIDTH, W).to(device)
    if num_gen_frames is not None:
        num_frame_red = L - num_gen_frames
        log.info(f"  Number of input frames (red): {num_frame_red}")
        log.info(f"  Number of generated frames (green): {num_gen_frames}")
    else:
        log.info("  All frames will be white (num_gen_frames is None)")

    step_size = W // L
    log.info(f"  Step size for color edge: {step_size}")

    # Rest of the function remains the same
    for i in range(L):
        if num_gen_frames is not None:
            if i < num_frame_red:
                # Add red color for input frames
                color_edge[
                    :, i, int(_COLOR_EDGE_WIDTH / 2) : _COLOR_EDGE_WIDTH, i * step_size : (i + 1) * step_size
                ] = (torch.tensor([1.0, 0.0, 0.0]).to(device).view(3, 1, 1))
            else:
                # Add green color for generated frames
                color_edge[
                    :, i, int(_COLOR_EDGE_WIDTH / 2) : _COLOR_EDGE_WIDTH, i * step_size : (i + 1) * step_size
                ] = (torch.tensor([0.0, 1.0, 0.0]).to(device).view(3, 1, 1))
        else:
            # Add white color for all frames
            color_edge[:, i, int(_COLOR_EDGE_WIDTH / 2) : _COLOR_EDGE_WIDTH, i * step_size : (i + 1) * step_size] = (
                torch.tensor([1.0, 1.0, 1.0]).to(device).view(3, 1, 1)
            )

        if pixel_chunk_duration is not None and i % pixel_chunk_duration == 0:
            # Add yellow color for chunk start
            color_edge[:, i, 0 : int(_COLOR_EDGE_WIDTH / 2), i * step_size : (i + 1) * step_size] = (
                torch.tensor([1.0, 1.0, 0.0]).to(device).view(3, 1, 1)
            )
    return color_edge


def get_num_gen_tokens(task_condition, latent_context_t_size, latent_shape):
    log.info(f"latent_shape: {latent_shape}")  # [L, 24, 40]
    T, H, W = latent_shape  # Number of latent tokens in time, height, width dimension

    if task_condition == "text_and_first_bov_token":
        num_gen_tokens = np.prod([T, H, W])
    elif task_condition in ["text_and_first_gt_token", "text_and_first_random_token"]:
        num_gen_tokens = np.prod([T, H, W]) - 1
    else:
        # Condition on latent_context_t_size tokens and generate T-latent_context_t_size tokens
        num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))
    return num_gen_tokens


def try_compute_num_frames(
    num_gen_tokens, latent_chunk_duration, pixel_chunk_duration, latent_shape, num_chunks_to_generate=1
):
    """Given the number of tokens to generate, the latent shape, and the tokenizer config,
    compute the number of pixel frames corresponding to it.

    """
    t = num_gen_tokens // (latent_shape[1] * latent_shape[2])
    if t % latent_chunk_duration == 0:
        num_frame = t // latent_chunk_duration * pixel_chunk_duration
    else:
        # The first chunk is not complete
        frame_for_complete_chunk = t // latent_chunk_duration * pixel_chunk_duration
        pixel_duration_per_latent = (pixel_chunk_duration - 1) // (latent_chunk_duration - 1)
        frame_for_first_chunk = t % latent_chunk_duration * pixel_duration_per_latent
        num_frame = frame_for_first_chunk + frame_for_complete_chunk
    log.info(
        f"round num_frame: {num_frame}; given num_gen_tokens: {num_gen_tokens * num_chunks_to_generate}; "
        f"latent_shape: {latent_shape}; T: {t}; "
        f"latent_chunk_duration: {latent_chunk_duration}; chunk_size: {pixel_chunk_duration}"
    )
    num_cond_frames = latent_shape[0] // latent_chunk_duration * pixel_chunk_duration - num_frame
    return num_frame, num_cond_frames
