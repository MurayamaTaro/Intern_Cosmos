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

import json
import os
from collections import OrderedDict
from enum import Enum

import torch
from typing import Dict, Any
from cosmos_predict1.utils import log
from typing import Optional

class CustomizationType(Enum):
    LORA = 1
    REPLACE = 2

    @classmethod
    def from_value(cls, value):
        """Convert both int and str to the corresponding enum."""
        if isinstance(value, str):
            value = value.lower()
            if value == "lora":
                return cls.LORA
            elif value == "replace":
                return cls.REPLACE
            elif value == "":
                return None
            else:
                raise ValueError("Customization type must be lora or replace")
        raise TypeError("CustomizationType must be specified as a string.")


class ApplyMode(Enum):
    CLEAR = 1
    ADD = 2


"""
A class for managing and applying 2 customizations types (LORA and REPLACE) to models, with support for two apply modes (CLEAR, ADD).

Apply Modes:
- CLEAR: Restore base model weights, then apply customizations.
- ADD: Apply customizations on top of existing customized weights.

Customization Types:
1. LORA:
   Naming:
   - Lora layers use a naming convention with dot or dash separators (e.g., `a.b.c.0.weight` or `a-b-c-0-weight`).
   - Modified blocks in Lora layers are appended with _lora and a new "net" block follows them (eg. a.b.c.0.weight [base]-> a.b.c_lora.net.0.weight [lora])
   - Lora customizations involve up (1) and down (0) tensors for each layer.
        eg: net.blocks.block0.blocks.0.block.attn.to_q_lora.net.1.weight (lora up)    -> net.blocks.block0.blocks.0.block.attn.to_q.0.weight (base model)
        net-blocks-block0-blocks-0-block-attn-to_q_lora-net-0-weight (lora down)

   Operations
   - Mode CLEAR: Restore base model weights, then add `scale * (u x d)` to base model weights.
   - Mode ADD: Add `scale * (u x d)` to the existing weights.

2. REPLACE:
    Naming
   - Replace specific layers with new weights while preserving the same layer names as the base model.

   Operations
   - Mode CLEAR: Restore base weights and replace with new checkpoint weights.
   - Mode ADD: Replace weights but preserve previously replaced values.

The class handles applying and bookkeeping these customizations per network.
"""


class CustomizationManager:
    def __init__(self, model: Optional[torch.nn.Module] = None, base_model_dict: Optional[OrderedDict] = None):
        # layer_name->tensor dict that stores the base model layer for any layer that is modified
        self.base_layer_cache = {}
        self.model = model
        assert model is not None or base_model_dict is not None, "model or base_model_dict must be provided"
        if model is None:
            self.model_state_dict = base_model_dict
        else:
            assert base_model_dict is None, "model and base_model_dict cannot both be provided"
            self.model_state_dict = model.state_dict()
            if model.state_dict().get("model", None):
                self.model_state_dict = model.state_dict()["model"]
        # Use to keep track of the last customization applied
        self.last_customization_meta_id = None

    def get_control_params(self, control_file: str) -> tuple[CustomizationType, float]:
        if not os.path.exists(control_file):
            raise Exception(f"Could not find control file: {control_file}")
        with open(control_file, "r") as control_file:
            data = json.load(control_file)
            try:
                customization_type_str = data["customization_type"]
                scale = float(data.get("scale", 1.0))
            except Exception:
                raise Exception(f"Could not find customization_type or scale in control file: {control_file}")

        if customization_type_str.lower() == "lora":
            customization_type = CustomizationType.LORA
        elif customization_type_str.lower() == "replace":
            customization_type = CustomizationType.REPLACE
        else:
            raise Exception(f"Invalid customization type: {customization_type_str}, expected LORA or REPLACE")
        return customization_type, scale

    def set_customization_type(self, customization_type: CustomizationType):
        if not isinstance(customization_type, CustomizationType):
            raise Exception(f"Invalid customization type: {customization_type}")
        self.customization_type = customization_type

    def set_apply_mode(self, apply_mode: str):
        if apply_mode == "add":
            self.apply_mode = ApplyMode.ADD
        elif apply_mode == "clear":
            self.apply_mode = ApplyMode.CLEAR
        else:
            raise Exception(f"Invalid apply mode: {apply_mode}")

    def _load_checkpoint_from_path(self, path: str) -> OrderedDict:
        return torch.load(path)

    def _get_base_name_from_lora_name(self, lora_name: str) -> str:
        base_name = lora_name.replace("_lora", "")
        base_name += ".0.weight"
        if base_name == lora_name:
            raise Exception(f"Lora layer name {lora_name} does not have the _lora suffix")

        log.debug(f"Converted layer name: lora name {lora_name} to base name {base_name}")
        return base_name

    def restore_weights(self) -> torch.nn.Module:
        # dry run to check that all the layers in the cache can be patched back into the model
        for base_layer_name, base_layer_val in self.base_layer_cache.items():
            if base_layer_name not in self.model_state_dict:
                raise Exception(
                    f"Could not restore to base checkpoint: cached layer {base_layer_name} not found in base checkpoint"
                )

            if self.model_state_dict[base_layer_name].shape != base_layer_val.shape:
                raise Exception(
                    f"Cannot restore to base: cached layer shape {base_layer_val.shape} does not match base layer shape {self.model_state_dict[base_layer_name].shape}"
                )

        # dry run passed, restore the weights
        self.model.refit_weights(self.base_layer_cache)
        self.base_layer_cache = {}
        return self.model

    def apply_customization(
        self,
        customization_path: str,
        apply_mode: str,
        control_file: str,
        customization_meta_id: tuple[str, dict[str, str], str],
    ) -> torch.nn.Module:
        if self.last_customization_meta_id == customization_meta_id:
            log.info(f"Customization {customization_meta_id} is already applied. Skipping ...")
            return self.model
        else:
            log.info(f"Applying customization_meta_id {customization_meta_id}")
        checkpoint = self._load_checkpoint_from_path(customization_path)
        self.set_apply_mode(apply_mode)
        customization_type, self.scale = self.get_control_params(control_file)
        self.set_customization_type(customization_type)
        log.info(f"type: {self.customization_type}, apply mode: {self.apply_mode}")

        # restore base and apply customization onto base (ie a clear-load)
        if self.apply_mode == ApplyMode.CLEAR:
            self.model = self.restore_weights()
            log.info("Restored base weights")

        # dry-run to check that the new checkpoint can be applied: includes checking shapes and layer names
        # if dry-run passes, the new weights to be refit are calculated - based on customization type - and returned
        weights_dict = {}
        if self.customization_type == CustomizationType.LORA:
            log.info("Applying LORA customization")
            weights_dict = self._dry_run_lora(checkpoint)

        elif self.customization_type == CustomizationType.REPLACE:
            log.info("Applying REPLACE customization")
            weights_dict = self._dry_run_replace(checkpoint)

        else:
            raise Exception(f"No implementation for customization type: {self.customization_type}")

        # dry-run passed, apply the customization and cache the changed layers
        for layer_name in weights_dict:
            log.debug(f"Cached customization layer {layer_name}")
            if layer_name not in self.base_layer_cache:
                self.base_layer_cache[layer_name] = self.model_state_dict[layer_name].clone()
        self.model.refit_weights(weights_dict)
        self.last_customization_meta_id = customization_meta_id

        return self.model

    def get_customized_weights(self, base_model_dict: OrderedDict[str, torch.Tensor], partial_checkpoint: Dict[str, Any], customization_type: CustomizationType, scale: float):
        """
            Get the new weights after applying the customization. Returns the new weights but does not apply them to the model.
        """
        self.__init__(model=None, base_model_dict=base_model_dict)
        self.set_apply_mode("clear")
        self.scale = scale
        if customization_type.lower() == "lora":
            self.set_customization_type(CustomizationType.LORA)
        elif customization_type.lower() == "replace":
            self.set_customization_type(CustomizationType.REPLACE)
        else:
            raise Exception(f"Invalid customization type: {customization_type}")
        log.info(f"type: {self.customization_type}, apply mode: {self.apply_mode}")
        # dry-run to check that the new checkpoint can be applied: includes checking shapes and layer names
        # if dry-run passes, the new weights to be refit are calculated - based on customization type - and returned
        weights_dict = {}
        if self.customization_type == CustomizationType.LORA:
            log.info("Applying LORA customization")
            weights_dict = self._dry_run_lora(partial_checkpoint)

        elif self.customization_type == CustomizationType.REPLACE:
            log.info("Applying REPLACE customization")
            weights_dict = self._dry_run_replace(partial_checkpoint)

        else:
            raise Exception(f"No implementation for customization type: {self.customization_type}")

        # Copy base_model_dict and update with the modified layers
        output_dict = base_model_dict.copy()
        for layer_name in weights_dict:
            output_dict[layer_name] = weights_dict[layer_name]

        return output_dict
         

    def _dry_run_lora(self, checkpoint: OrderedDict) -> dict[str, torch.Tensor]:
        lora_layers_dict = {}
        # pair all the loras with the same prefix name together (the prefix name for a.b.c_lora.net.0.weight is a.b.c_lora)
        #                                                       (the prefix name for a-b.c_lora-net-0-weight is a.b.c_lora)
        for layer_name in checkpoint:
            if "_lora" in layer_name and "_extra_state" not in layer_name:
                lora_layer_split = layer_name.split(".")
                if len(lora_layer_split) == 1:
                    lora_layer_split = layer_name.split("-")
                lora_layer_prefix = ".".join(lora_layer_split[:-3])

                # get the "index" representing either an up tensor or down tensor
                lora_layer_index = lora_layer_split[-2]
                if not lora_layer_index.isdigit():
                    raise Exception(
                        f"Could not apply customization: the lora layer doesn't follow the expected naming scheme: {layer_name}"
                    )
                lora_layer_type = "down" if int(lora_layer_index) == 0 else "up" if int(lora_layer_index) == 1 else -1
                if lora_layer_type == -1:
                    raise Exception(
                        f"Could not apply customization: the lora layer doesn't follow the expected naming scheme: {layer_name}"
                    )
                if lora_layer_prefix not in lora_layers_dict:
                    lora_layers_dict[lora_layer_prefix] = {}
                if lora_layer_type in lora_layers_dict[lora_layer_prefix]:
                    raise Exception(
                        f"Could not apply customization: the lora layer has multiple tensors of the same type: {layer_name}"
                    )
                lora_layers_dict[lora_layer_prefix][lora_layer_type] = checkpoint[layer_name]
            elif "_extra_state" not in layer_name:
                raise Exception(
                    f"Could not apply customization: the lora layer doesn't follow the expected naming scheme: {layer_name}"
                )

        # verify that for each lora layer, we have:
        #    - corresponding up and down tensors
        #    - after converting from the lora naming convention to the standard naming convention, ensure that a base layer with the modified name exists
        #    - the shape of u cross d matches the base layer shape
        weights_dict = {}
        for lora_layer_name in lora_layers_dict:
            if "down" not in lora_layers_dict[lora_layer_name] or "up" not in lora_layers_dict[lora_layer_name]:
                raise Exception(
                    f"Could not apply customization: lora layer {lora_layer_name} does not have both up and down tensors"
                )

            base_layer_name = self._get_base_name_from_lora_name(lora_layer_name)
            if base_layer_name not in self.model_state_dict:
                raise Exception(
                    f"Could not apply customization: new layer {base_layer_name} not found in base checkpoint"
                )

            lora_layer_shape = [
                lora_layers_dict[lora_layer_name]["up"].shape[0],
                lora_layers_dict[lora_layer_name]["down"].shape[1],
            ]
            if (
                self.model_state_dict[base_layer_name].shape[0] != lora_layer_shape[0]
                or self.model_state_dict[base_layer_name].shape[1] != lora_layer_shape[1]
            ):
                raise Exception(
                    f"Could not apply customization: input layer shape for {base_layer_name} {lora_layer_shape} does not match base model {self.model_state_dict[base_layer_name].shape}"
                )
            mult = self.scale * torch.matmul(
                lora_layers_dict[lora_layer_name]["up"], lora_layers_dict[lora_layer_name]["down"]
            )
            device = mult.device
            weights_dict[base_layer_name] = self.model_state_dict[base_layer_name].to(device) + mult

        return weights_dict

    def _dry_run_replace(self, checkpoint: OrderedDict) -> dict[str, torch.Tensor]:
        weights_dict = {}
        for layer_name, layer_val in checkpoint.items():
            if layer_name not in self.model_state_dict:
                raise Exception(f"Could not apply customization: new layer {layer_name} not found in base checkpoint")
            if layer_val.shape != self.model_state_dict[layer_name].shape:
                raise Exception(
                    f"Could not apply customization: input layer shape for {layer_name} {layer_val.shape} does not match base model {self.model_state_dict[layer_name].shape}"
                )
            weights_dict[layer_name] = layer_val
        return weights_dict
