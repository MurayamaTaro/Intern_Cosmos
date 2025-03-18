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

"""
    This file contains a basic configuration for video2video experiments.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict1.autoregressive.configs.base.model_config import create_video2world_model
from cosmos_predict1.autoregressive.configs.base.model_parallel import create_model_parallel_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


"""
Train 1B model from scratch with 1 TP and 1 CP, pytorch backend, mock data.
Usage:
torchrun --nproc_per_node=1 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_1b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""


VIDEO2WORLD_1B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "mock_video"},
            {
                "override /callbacks": [
                    "basic",
                    "video_teacher_forcing",
                    "video_partial_tokens",
                    # "output_monitor",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /optimizer": "fused_adamw"},
            {"override /scheduler": "warmup_cosine_lr"},
            "_self_",
        ],
        model=create_video2world_model(
            tensor_model_parallel_size=1,
            model_size="1b",
            backend="pytorch",
            fsdp_enabled=False,
            model_family="cosmos",
            pixel_chunk_duration=9,
            num_video_frames=36,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
        ),
        trainer=dict(
            max_iter=50000,
            grad_accum_iter=1,
            grad_scaler_args=dict(enabled=False),
            run_validation=False,  # No need for validation as epoch <= 1
            distributed_parallelism="ddp",
        ),
        model_parallel=create_model_parallel_config(),
        job=dict(group="debug", name="basic_video2world_1b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
    flags={"allow_objects": True},
)
log.info("Registering experiment video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
    node=VIDEO2WORLD_1B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK,
)


"""
   Train 4B model from scratch with 1 TP and 1 CP, pytorch backend, mock data.
   Usage:
   torchrun --nproc_per_node=1 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""
VIDEO2WORLD_4B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
            "_self_",
        ],
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=1,
            shard_checkpoint=True,
            batch_size=1,
            pixel_chunk_duration=9,
            num_video_frames=36,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
        ),
        checkpoint=dict(
            load_path="",
            load_training_state=False,
            strict_resume=True,
            save_iter=1000,
        ),
        job=dict(group="debug", name="basic_video2world_4b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
)
log.info("Registering experiment video2world_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
    node=VIDEO2WORLD_4B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK,
)

"""
   Finetune 4B model with 1 TP and 1 CP, pytorch backend, mock data.
   Usage:
   torchrun --nproc_per_node=1 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""
VIDEO2WORLD_FT_4B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
            "_self_",
        ],
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=1,
            shard_checkpoint=True,
            batch_size=1,
            pixel_chunk_duration=9,
            num_video_frames=36,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Predict1-4B/model.pt",
            load_training_state=False,
            strict_resume=True,
            save_iter=1000,
        ),
        job=dict(group="debug", name="basic_video2world_ft_4b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
)
log.info("Registering experiment video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
    node=VIDEO2WORLD_FT_4B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK,
)


"""
   Finetune 4B model with 4 TP and 1 CP, pytorch backend, bridge data, frame 33, chunk 33.
   Usage:
   torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_ft_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""
VIDEO2WORLD_FT_4B_TP4_CP1_PT_DDP_FRAME33_CHUNK33_BRIDGE: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
            {"override /data_train": "bridge_video"},
            "_self_",
        ],
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=4,
            shard_checkpoint=True,
            batch_size=1,
            pixel_chunk_duration=33,
            num_video_frames=33,
            video_height=640,
            video_width=848,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
            training_type="video_to_video",
            pad_to_multiple_of=1,
        ),
        job=dict(group="debug", name="basic_video2world_ft_4b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
)
log.info("Registering experiment video2world_ft_4b_tp1_cp1_pt_ddp_frame33_chunk33_bridge")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_ft_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge",
    node=VIDEO2WORLD_FT_4B_TP4_CP1_PT_DDP_FRAME33_CHUNK33_BRIDGE,
)


"""
   Train from scratch 4B model with 4 TP and 1 CP, pytorch backend, bridge data, frame 33, chunk 33.
   Usage:
   torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""
VIDEO2WORLD_4B_TP4_CP1_PT_DDP_FRAME33_CHUNK33_BRIDGE: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
            {"override /data_train": "bridge_video"},
            "_self_",
        ],
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=4,
            shard_checkpoint=True,
            batch_size=1,
            pixel_chunk_duration=33,
            num_video_frames=33,
            video_height=640,
            video_width=848,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
            training_type="video_to_video",
            pad_to_multiple_of=1,
        ),
        checkpoint=dict(
            load_path="",
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
        ),
        job=dict(group="debug", name="basic_video2world_4b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
)
log.info("Registering experiment video2world_4b_tp1_cp1_pt_ddp_frame33_chunk33_bridge")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge",
    node=VIDEO2WORLD_4B_TP4_CP1_PT_DDP_FRAME33_CHUNK33_BRIDGE,
)


# TEACHER FORCING DOES NOT REACH 100%

"""
   Train 4B model from scratch with 4 TP and 1 CP, transformer engine backend, mock data.
   Usage:
   torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_4b_tp4_cp1_te_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""
VIDEO2WORLD_4B_TP4_CP1_TE_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
            "_self_",
        ],
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="transformer_engine",
            tensor_model_parallel_size=4,
            shard_checkpoint=True,
            batch_size=1,
            pixel_chunk_duration=9,
            num_video_frames=36,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
        ),
        checkpoint=dict(load_path=""),
        job=dict(group="debug", name="basic_video2world_4b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
)
log.info("Registering experiment video2world_4b_tp4_cp1_te_ddp_frame36_chunk9_mock")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_4b_tp4_cp1_te_ddp_frame36_chunk9_mock",
    node=VIDEO2WORLD_4B_TP4_CP1_TE_DDP_FRAME36_CHUNK9_MOCK,
)


"""
   Train 4B model from scratch with 4 TP and 1 CP, pytorch backend, mock data.
   Usage:
   torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
"""
VIDEO2WORLD_4B_TP4_CP1_PT_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
            "_self_",
        ],
        model=create_video2world_model(
            model_size="4b",
            model_family="cosmos",
            backend="pytorch",
            tensor_model_parallel_size=4,
            shard_checkpoint=True,
            batch_size=1,
            pixel_chunk_duration=9,
            num_video_frames=36,
            tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
            add_special_tokens=False,
        ),
        checkpoint=dict(load_path=""),
        job=dict(group="debug", name="basic_video2world_4b_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ),
)
log.info("Registering experiment video2world_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock")
cs.store(
    group="experiment",
    package="_global_",
    name="video2world_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock",
    node=VIDEO2WORLD_4B_TP4_CP1_PT_DDP_FRAME36_CHUNK9_MOCK,
)


# TEACHER FORCING NOT REACH 100%
# """
#    Finetune 4B model with 4 TP and 1 CP, pytorch backend, mock data.
#    Usage:
#    torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_ft_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
# """
# VIDEO2WORLD_FT_4B_TP4_CP1_PT_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
#     dict(
#         defaults=[
#             "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
#             "_self_",
#         ],
#         model=create_video2world_model(
#             model_size="4b",
#             model_family="cosmos",
#             backend="pytorch",
#             tensor_model_parallel_size=4,
#             shard_checkpoint=True,
#             batch_size=1,
#             pixel_chunk_duration=9,
#             num_video_frames=36,
#             tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
#             add_special_tokens=False,


#         ),
#         checkpoint=dict(
#             load_path="checkpoints/Cosmos-Predict1-4B/model.pt",
#             load_training_state=False,
#             strict_resume=True,
#             save_iter=1000,
#         ),
#         job=dict(group="debug", name="basic_video2world_ft_4b_tp4_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
#     ),
# )
# log.info("Registering experiment video2world_ft_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock")
# cs.store(
#     group="experiment",
#     package="_global_",
#     name="video2world_ft_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock",
#     node=VIDEO2WORLD_FT_4B_TP4_CP1_PT_DDP_FRAME36_CHUNK9_MOCK,
# )

# TEACHER FORCING NOT REACH 100%
# """
#    Finetune 4B model with 4 TP and 1 CP, transformer engine backend, mock data.
#    Usage:
#    torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_ft_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
# """
# VIDEO2WORLD_FT_4B_TP4_CP1_TE_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
#     dict(
#         defaults=[
#             "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
#             "_self_",
#         ],
#         model=create_video2world_model(
#             model_size="4b",
#             model_family="cosmos",
#             backend="transformer_engine",
#             tensor_model_parallel_size=4,
#             shard_checkpoint=True,
#             batch_size=1,
#             pixel_chunk_duration=9,
#             num_video_frames=36,
#             tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
#             add_special_tokens=False,


#         ),
#         checkpoint=dict(
#             load_path="checkpoints/Cosmos-Predict1-4B/model.pt",
#             load_training_state=False,
#             strict_resume=True,
#             save_iter=1000,
#         ),
#         job=dict(group="debug", name="basic_video2world_ft_4b_tp4_te_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
#     ),
# )
# log.info("Registering experiment video2world_ft_4b_tp4_cp1_te_ddp_frame36_chunk9_mock")
# cs.store(
#     group="experiment",
#     package="_global_",
#     name="video2world_ft_4b_tp4_cp1_te_ddp_frame36_chunk9_mock",
#     node=VIDEO2WORLD_FT_4B_TP4_CP1_TE_DDP_FRAME36_CHUNK9_MOCK,
# )


# TEACHER FORCING DOES NOT REACH 100%

# """
#    Train 4B model from scratch with 4 TP and 1 CP, pytorch backend, mock data.
#    Usage:
#    torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_4b_tp4_cp1_te_ddp_frame36_chunk9_mock job.name=local_debug_vid2world_4b job.group=debug trainer.callbacks.vid_sampling_tf.every_n=2
# """
# VIDEO2WORLD_4B_TP4_CP1_PT_DDP_FRAME36_CHUNK9_MOCK: LazyDict = LazyDict(
#     dict(
#         defaults=[
#             "/experiment/video2world_1b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
#             "_self_",
#         ],
#         model=create_video2world_model(
#             model_size="4b",
#             model_family="cosmos",
#             backend="pytorch",
#             tensor_model_parallel_size=4,
#             shard_checkpoint=True,
#             batch_size=1,
#             pixel_chunk_duration=9,
#             num_video_frames=36,
#             tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
#             add_special_tokens=False,

#         ),
#         checkpoint=dict(
#             # load_path="checkpoints/Cosmos-Predict1-4B/model.pt",
#             load_path="",
#             load_training_state=False,
#             # strict_resume=True,
#             save_iter=1000,
#         ),
#         job=dict(group="debug", name="basic_video2world_4b_tp4_pt_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
#     ),
# )
# log.info("Registering experiment video2world_4b_tp4_pt_mockdata_${now:%Y-%m-%d}_${now:%H-%M-%S}")
# cs.store(
#     group="experiment",
#     package="_global_",
#     name="video2world_4b_tp4_cp1_pt_ddp_frame36_chunk9_mock",
#     node=VIDEO2WORLD_4B_TP4_CP1_PT_DDP_FRAME36_CHUNK9_MOCK,
# )
