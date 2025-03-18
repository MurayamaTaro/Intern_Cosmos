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

"""Dataset config class."""

from typing import Union

import attrs

from cosmos_predict1.utils.config import make_freezable


@make_freezable
@attrs.define(slots=False)
class OverfitDatasetConfig:
    # """
    # A class to hold overfitting dataset.

    # Args:
    #     data_key (str): The input key from data_dict.
    # """
    data_key: str = "video"
    batch_size: Union[int, None] = None
    num_video_frames: Union[int, None] = None
    single_data: bool = False  # always return the first data
    # Raw video pixel dimension, input video will be resized to this dimension
    video_height: int = 384
    video_width: int = 640


@make_freezable
@attrs.define(slots=False)
class BridgeDatasetConfig:
    """
    Args:
        dataset_dir (str): Base path to the dataset directory
        sequence_interval (int): Interval between sampled frames in a sequence
        num_frames (int): Number of frames to load per sequence
        video_size (list): Target size [H,W] for video frames
        start_frame_interval (int): Interval between starting frames of sequences
    """

    dataset_dir: str = "assets/example_training_data_bridge/"
    sequence_interval: int = 1
    num_frames: int = 33
    video_size: list[int, int] = [640, 848]
    start_frame_interval: int = 1
