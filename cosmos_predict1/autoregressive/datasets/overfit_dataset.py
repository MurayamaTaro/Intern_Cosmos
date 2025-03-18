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

import torch
from einops import rearrange

from cosmos_predict1.autoregressive.configs.base.dataset import OverfitDatasetConfig
from cosmos_predict1.utils import log


def get_mock_video_data_batch(length: int = 36, video_height: int = 384, video_width: int = 640):
    video_data_batch = {}
    n = 2
    # generate video non-random values are between -1 and 1
    video_data_batch["video"] = torch.zeros(n, 3, length, video_height, video_width, dtype=torch.bfloat16)
    video_data_batch["video"][:, :, :, 0:100, :] = 1
    video_data_batch["video"][-1, :, :, 0:100, 100:200] = -1
    video_data_batch["padding_mask"] = torch.zeros(n, 1, video_height, video_width, dtype=torch.bfloat16)
    video_data_batch["padding_mask"][:, :, -20:, -20:] = 1
    video_data_batch["fps"] = torch.tensor([25] * n, dtype=torch.bfloat16)
    video_data_batch["num_frames"] = torch.tensor([length] * n, dtype=torch.bfloat16)
    video_data_batch["chunk_index"] = torch.tensor([1] * n, dtype=torch.bfloat16)
    video_data_batch["image_size"] = torch.tensor([[video_height, video_width, 360, 640]] * n, dtype=torch.bfloat16)
    video_data_batch["caption"] = (
        [
            """
    This video captures a beautiful sunset over a calm ocean. The sky is painted with hues of orange, pink, and purple, while the waves gently roll in. The sun is setting, casting a warm glow on the water and the surrounding landscape.
    In the distance, a small boat is sailing peacefully.
    """,
        ]
        * n
    )
    return video_data_batch


class OverfitDataset:
    """
    Dataset for returning the same batch. This is useful for overfitting experiments
    """

    def __init__(self, config: OverfitDatasetConfig):
        super().__init__()
        if config.data_key == "video":
            self.data_batch = get_mock_video_data_batch(
                length=config.num_video_frames,
                video_height=config.video_height,
                video_width=config.video_width,
            )
        else:
            raise ValueError(f"Data key {config.data_key} not supported")
        self.config = config
        self.single_data = config.single_data  # Return the same data batch for each iteration
        self.counter = 0
        log.info(f"Loaded data {config.data_key} mock data, always return same data (single_data)={self.single_data}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.config.batch_size is not None:
            total_num = self.data_batch[self.config.data_key].shape[0]
            if self.config.batch_size > total_num:
                log.warning(
                    f"Batch size {self.config.batch_size} is greater than the total number of samples {total_num}. Returning the full dataset."
                )
                data_batch_cur = self.data_batch
            else:
                # Subset the data batch
                data_batch_cur = {}
                start = (self.counter * self.config.batch_size) % (total_num - self.config.batch_size + 1)
                for key, value in self.data_batch.items():
                    if isinstance(value, list) or isinstance(value, torch.Tensor):
                        data_batch_cur[key] = value[start : start + self.config.batch_size]
                    else:
                        data_batch_cur[key] = value
        else:
            # Batch size is None, return the full dataset
            data_batch_cur = self.data_batch

        if "video" in self.config.data_key and self.config.num_video_frames is not None:
            # Truncate the video frames
            data_batch_cur[self.config.data_key] = data_batch_cur[self.config.data_key][
                :, :, : self.config.num_video_frames
            ]  # B, D, L, H, W
            data_batch_cur["num_frames"] = self.config.num_video_frames
            # Interpolate the tensor to the desired size
            b, d, l, h, w = data_batch_cur[self.config.data_key].shape
            data_batch_cur[self.config.data_key] = torch.nn.functional.interpolate(
                rearrange(data_batch_cur[self.config.data_key], "b d l h w -> (b l) d h w"),
                size=(self.config.video_height, self.config.video_width),
                mode="bilinear",
                align_corners=False,
            )
            data_batch_cur[self.config.data_key] = rearrange(
                data_batch_cur[self.config.data_key], "(b l) d h w -> b d l h w", b=b, l=l
            )

        if not self.single_data:
            self.counter += 1  # Increment the counter
        return data_batch_cur

    def __len__(self) -> int:
        if self.single_data:
            return 1
        elif self.config.batch_size is not None:
            return len(self.data_batch[self.config.data_key]) // self.config.batch_size
        else:
            return 1
