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

import argparse
import importlib
import os

from loguru import logger as logging
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from omegaconf import OmegaConf

from cosmos_predict1.checkpointer.tp import Checkpointer as TensorParallelCheckpointer
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.config_helper import get_config_module, override
from cosmos_predict1.utils.lazy_config import instantiate
from cosmos_predict1.utils.lazy_config.lazy import LazyConfig

# from cosmos_predict1.utils.launch import log_reproducible_setup
# from cosmos_predict1.utils.one_logger.one_logger_override_utils import override_one_logger_callback
# from cosmos_predict1.utils.one_logger.one_logger_utils import get_one_logger, one_logger_is_initialized
# from projects.cosmos.diffusion.v1.checkpointers.s3_fast_tp import Checkpointer as S3FastTensorParallelCheckpointer

"""
The training entry script for the Edify-World-A project. Works for both DDP and FSDP training.
Example usage:
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=4 --master_port=2001 projects/cosmos/ar/v1/train.py
"""


@misc.timer("instantiate LLM")
def instantiate_model(config, trainer) -> None:
    model_parallel_cuda_manual_seed(config.trainer.seed)
    # if config.model["model_config"].fsdp_enabled:
    #     # As FSDP is enabled, we need to pass the FSDP checkpointer to the FSDP model constructor
    #     log.critical("FSDP enabled")
    #     config.model["fsdp_checkpointer"] = trainer.checkpointer
    model = instantiate(config.model)
    # if not config.model["model_config"].set_parallel_mode:
    #     misc.set_random_seed(seed=config.trainer.seed, by_rank=True)

    return model


@logging.catch(reraise=True)
def launch(config, args: argparse.Namespace) -> None:
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Setup the miscellaneous stuff for reproducibility.
    # log_reproducible_setup(config, args)
    # Create the model
    model = instantiate_model(config, trainer)

    # if args.cluster is not None and args.cluster.startswith("aws"):
    #     if isinstance(trainer.checkpointer, TensorParallelCheckpointer):
    #         del trainer.checkpointer
    #         trainer.checkpointer = S3FastTensorParallelCheckpointer(
    #             config.checkpoint, config.job, callbacks=trainer.callbacks
    #         )
    #         log.critical(f"[cluster={args.cluster}] Replaced TP checkpointer with the S3-Fast TP Checkpointer")

    model.on_model_init_end()
    # # get OneLoggerUtils object
    # one_logger = get_one_logger()  # globally unique one_logger is initialized in trainer
    # # Create the dataloaders.
    # if one_logger_is_initialized():
    #     one_logger.on_dataloader_init_start()
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    # if one_logger_is_initialized():
    #     one_logger.on_dataloader_init_end()
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config", default="projects.cosmos.ar.v1.configs.train_openhermes", help="Path to the config file"
    )
    parser.add_argument("--cluster", default=None, help="Cluster name")
    parser.add_argument(
        "opts",
        help="""Modify config options at the end of the command. For Yacs configs, use
                space-separated "PATH.KEY VALUE" pairs.
                For python-based LazyConfig, use "path.key=value".
                """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    config = importlib.import_module(get_config_module(args.config)).make_config()
    config = override(config, args.opts)
    # config = override_one_logger_callback(config)
    if args.dryrun:
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(OmegaConf.to_yaml(OmegaConf.load(f"{config.job.path_local}/config.yaml")))
    else:
        # Launch the training job.
        launch(config, args)
