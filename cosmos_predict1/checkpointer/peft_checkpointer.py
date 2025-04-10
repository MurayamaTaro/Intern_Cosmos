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
from cosmos_predict1.checkpointer.ddp import Checkpointer as DDPCheckpointer
from cosmos_predict1.utils.model import Model
from typing import Dict, Set, Any
import os
import torch

from megatron.core import parallel_state
from cosmos_predict1.utils import distributed, log
from cosmos_predict1.checkpointer.safe_broadcast import broadcast_object
from torch.distributed import ProcessGroup, get_process_group_ranks
from typing import Optional, Dict, Any
from collections import OrderedDict
from cosmos_predict1.diffusion.training.utils.customization.customization_manager import CustomizationManager

class Checkpointer(DDPCheckpointer):
    """
    Checkpointer class for PEFT in distributed training.
    Note:
    - Fully Sharded Data Parallelism (FSDP) is not supported by this checkpointer.
    - Multi-GPU is not supported by this checkpointer.
    """
    KEYS_TO_SAVE = ["model", "merged_model", "partial_model", "optim", "scheduler", "trainer"]
    KEYS_TO_POSTFIX = {
        "model": "model",
        "merged_model": "merged",
        "partial_model": "partial",
        "optim": "optim",
        "scheduler": "scheduler",
        "trainer": "",
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.broadcast_via_filesystem:
            raise ValueError("self.broadcast_via_filesystem=False is not implemented for PEFT checkpointer.")
        self.base_model_state_dict = None # This will be instantiated once and assumed to remain constant across iterations

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: Model) -> str:
        """
        Overwrite the `add_type_postfix_to_checkpoint_path` function of the base class (DDP checkpointer)
        to load pre-trained model without any postfix.
        """
        checkpoint_path = super().add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
        checkpoint_path = checkpoint_path.replace("model_model.pt", "model.pt")
        return checkpoint_path
    
    def load_broadcast_state_dict(self, checkpoint_path: str, model: Model, resume_keys: Set) -> dict[str, Any]:
        """
        Load state_dict and broadcast.

        The main steps are:
        1. Download TP-rank-specific checkpoints for every GPU of DDP-rank 0 and CP-rank 0.
        2. Each rank loads its corresponding checkpoint from the local cache or receives it via broadcast.

        This approach ensures that each MP rank loads its specific part of the model, which is
        crucial for Model Parallelism where different parts of the model are distributed across
        multiple GPUs.

        When using Model Parallelism (e.g., Tensor Parallelism), the `broadcast_via_filesystem` option can
        be set to True. This allows each rank to load its specific checkpoint from the local filesystem
        instead of receiving it via network broadcast, which could be more efficient in some cases.

        For standard DDP without TP, `broadcast_via_filesystem` should remain False (default).

        Args:
            checkpoint_path (str): The base path of the checkpoint.
            model (Model): The model being loaded.
            resume_keys (Set): Set of keys to resume from the checkpoint.

        Returns:
            dict[str, Any]: A dictionary containing the loaded state for each resumed key.
        """
        state_dict = {}
        sorted_resume_keys = sorted(resume_keys)
        # Step 1: Download checkpoints for every GPU of DDP-rank 0 and CP-rank 0.
        if self.rank_dp_w_cp == 0:
            for key in sorted_resume_keys:
                _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                if os.path.exists(local_cache_path):
                    # If the local checkpoint exists, we can directly load it
                    self.print(f"Checkpoint is already in local cache: {local_cache_path}. Loading...")
                    _state_dict = torch.load(local_cache_path, map_location=lambda storage, loc: storage, weights_only=False)
                else:
                    # Pre-trained model is not in local cache, so we need to load it from the checkpoint path
                    self.print(f"Loading checkpoint from: {_ckpt_path}")
                    _state_dict = torch.load(_ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
                state_dict[key] = _state_dict
        
        # Ensure all ranks wait for the download to complete
        distributed.barrier()

        # Step 2: Broadcast checkpoint data
        log.info(
            "Start broadcasting checkpoint from the source rank to all other ranks in the same DDP group.",
            rank0_only=True,
        )
        for key in sorted_resume_keys:
            if self.broadcast_via_filesystem:
                # Load the checkpoint from the local filesystem for other ranks
                if self.rank_dp_w_cp != 0:
                    _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                    local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                    if os.path.exists(local_cache_path):
                        self.print(f"Loading checkpoint from: {local_cache_path}")
                        state_dict[key] = torch.load(local_cache_path, map_location=lambda storage, loc: storage, weights_only=False)
                    else:
                        self.print(f"Loading checkpoint from: {_ckpt_path}")
                        state_dict[key] = torch.load(
                            _ckpt_path, map_location=lambda storage, loc: storage, weights_only=False
                        )

            else:
                raise ValueError("self.broadcast_via_filesystem=False is not implemented for PEFT checkpointer.")

        return state_dict
    
    def generate_save_state_dict(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> Optional[Dict[str, Any]]:
        state_dict = super().generate_save_state_dict(
            model=model, optimizer=optimizer, scheduler=scheduler, grad_scaler=grad_scaler, iteration=iteration
        )

        # This logic does not impact _model.pt 
        if self.rank_dp_w_cp == 0:
            partial_model = self._filter_trainable_state_dict(model)["ema"]
            log.info(f"Partial model state dict has {len(partial_model)} keys")
            if self.base_model_state_dict is None:
                self.base_model_state_dict = self._filter_base_model_state_dict(model)
                log.info(f"Base model state dict has {len(self.base_model_state_dict)} keys")
            customization_manager = CustomizationManager(model=None, base_model_dict=self.base_model_state_dict)
            peft_control = model.config.peft_control
            customization_type, scale = peft_control["customization_type"], peft_control["scale"]
            merged_weights = customization_manager.get_customized_weights(self.base_model_state_dict, partial_model, customization_type, scale)
            state_dict["merged_model"] = merged_weights
            
        return state_dict
    
    def _filter_trainable_state_dict(self, model):
        """returns state dict with trainable parameters only - used for peft"""
        trainable_param_names = {name for name, param in model.model.named_parameters() if param.requires_grad}
        state_dict_keys = [k for k in model.state_dict().keys() if k in ["ema", "model", "trained_data_record"]]
        output_dict = OrderedDict({k: OrderedDict() for k in state_dict_keys})
        for state_dict_key in state_dict_keys:
            if state_dict_key == "trained_data_record":
                output_dict[state_dict_key] = model.state_dict()[state_dict_key]
                continue
            sub_state_dict = model.state_dict()[state_dict_key]
            for k, v in sub_state_dict.items():
                if k in trainable_param_names or k.replace("-", ".") in trainable_param_names:
                    output_dict[state_dict_key][k] = v
        return output_dict
    
    def _filter_base_model_state_dict(self, model):
        """returns state dict with base model parameters only - used for peft"""
        trainable_param_names = {name for name, param in model.model.named_parameters() if param.requires_grad}
        output_dict = OrderedDict()
        for k, v in model.state_dict()["model"].items():
            if k not in trainable_param_names and k.replace("-", ".") not in trainable_param_names:
                output_dict[k] = v
        return output_dict