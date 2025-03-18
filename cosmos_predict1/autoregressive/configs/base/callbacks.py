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

# from projects.cosmos.ar.v1.callbacks.image_sampling_teacher_forcing import ImageSamplingTeacherForcing
# from projects.cosmos.ar.v1.callbacks.output_monitor import OutputMonitor
# from projects.cosmos.ar.v1.callbacks.torch_compile import TorchCompile
# from projects.cosmos.ar.v1.callbacks.video_per_frame_loss import VideoPerFrameLoss
from cosmos_predict1.autoregressive.callbacks.video_sampling_partial_tokens import VideoSamplingPartialTokens
from cosmos_predict1.autoregressive.callbacks.video_sampling_teacher_forcing import VideoSamplingTeacherForcing
from cosmos_predict1.autoregressive.configs.inference.inference_config import TrainingSamplingConfig as SamplingConfig

# from projects.cosmos.diffusion.v1.callbacks.device_monitor import DeviceMonitor
# from projects.cosmos.diffusion.v1.callbacks.heart_beat import HeartBeat
# from projects.edify_image.v4.callbacks.dataloading_monitor import DetailedDataLoadingSpeedMonitor
from cosmos_predict1.callbacks.grad_clip import GradClip
from cosmos_predict1.utils.callback import ProgressBarCallback
from cosmos_predict1.utils.lazy_config import LazyCall as L

# from projects.edify_image.v4.callbacks.iter_speed import IterSpeed
# from projects.edify_image.v4.callbacks.norm_monitor import NormMonitor
# from projects.edify_image.v4.callbacks.param_count import ParamCount

BASIC_CALLBACKS = dict(
    progress_bar=L(ProgressBarCallback)(),
    grad_clip=L(GradClip)(clip_norm=1.0, fsdp_enabled="${model.model_config.fsdp_enabled}", model_key="model"),
    # norm_monitor=L(NormMonitor)(every_n=100, model_key="model", save_s3=False, log_stat_wandb=True),
    # iter_speed=L(IterSpeed)(
    #     every_n="${trainer.logging_iter}",
    #     save_s3="${upload_reproducible_setup}",
    #     save_s3_every_log_n=100,
    # ),
    # torch_compile=L(TorchCompile)(
    #     compile_after_iterations=4,
    #     compile_video_tokenizer_encode=False,
    #     video_tokenizer_encode_compile_dynamic_option=False,
    # ),
    # manual_gc=L(ManualGarbageCollection)(every_n=5),
)

VIDEO_TEACHER_FORCING_CALLBACK = dict(
    vid_sampling_tf=L(VideoSamplingTeacherForcing)(
        every_n=500,
        video_latent_shape="${model.model_config.video_latent_shape}",
        num_frames_to_display=4,
        save_folder="video_sampling_teacher_forcing",
    )
)

# IMAGE_TEACHER_FORCING_CALLBACK = dict(
#     img_sampling_tf=L(ImageSamplingTeacherForcing)(
#         every_n=500,
#         image_latent_shape="${model.model_config.image_latent_shape}",
#         save_folder="image_sampling_teacher_forcing",
#     )
# )

# LOW_PRECISION_CALLBACK = dict(
#     low_precision=L(LowPrecisionCallback)(trainer=PLACEHOLDER, config=PLACEHOLDER, update_iter=2)
# )

# OUTPUT_MONITOR_CALLBACK = dict(output_monitor=L(OutputMonitor)(every_n=1))

# VIDEO_PER_FRAME_LOSS_CALLBACK = dict(video_per_frame_loss=L(VideoPerFrameLoss)())

# JOB_MONITOR_CALLBACKS = dict(
#     param_count=L(ParamCount)(
#         save_s3="${upload_reproducible_setup}",
#     ),
#     heart_beat=L(HeartBeat)(
#         every_n=100,
#         update_interval_in_minute=20,
#         save_s3="${upload_reproducible_setup}",
#     ),
#     device_monitor=L(DeviceMonitor)(
#         every_n=100,
#         save_s3="${upload_reproducible_setup}",
#     ),
#     dataloader_speed=L(DetailedDataLoadingSpeedMonitor)(
#         every_n=100,
#         save_s3="${upload_reproducible_setup}",
#     ),
# )


def create_video_partial_sampling_callback(task_condition: str, latent_context_t_sizes: list = [1]):
    """Create a video partial token sampling callback by varying latent_context_t_sizes."""
    return dict(
        vid_sampling_partial_tokens=L(VideoSamplingPartialTokens)(
            every_n=5000,
            task_condition=task_condition,
            sampling_config=SamplingConfig(echo=True, temperature=1.0),
            video_latent_shape="${model.model_config.video_latent_shape}",
            latent_context_t_sizes=latent_context_t_sizes,
            save_folder="video_sampling_partial_tokens",
            iteration_early_test=5000,
        )
    )


# if we set latent_context_t_sizes=[1], we have one condition frame, which is image(video)-to-video
VIDEO_PARTIAL_TOKENS_CALLBACK = create_video_partial_sampling_callback(
    task_condition="video", latent_context_t_sizes=[1]
)


# def create_callback_group_video2video(task_condition, sampling=True, latent_context_t_sizes=[1]):
#     """Create a callback group for video2video experiments."""
#     callback_group = dict()
#     callback_group.update(BASIC_CALLBACKS)
#     callback_group.update(VIDEO_TEACHER_FORCING_CALLBACK)
#     if sampling:
#         callback_group.update(
#             create_video_partial_sampling_callback(
#                 task_condition=task_condition, latent_context_t_sizes=latent_context_t_sizes
#             )
#         )
#     callback_group.update(LOW_PRECISION_CALLBACK)
#     callback_group.update(OUTPUT_MONITOR_CALLBACK)
#     callback_group.update(VIDEO_PER_FRAME_LOSS_CALLBACK)
#     callback_group.update(JOB_MONITOR_CALLBACKS)
#     return callback_group
