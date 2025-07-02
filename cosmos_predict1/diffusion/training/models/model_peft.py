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

from typing import Dict, Type, TypeVar

from cosmos_predict1.diffusion.training.models.extend_model import ExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model import DiffusionModel as VideoDiffusionModel
from cosmos_predict1.diffusion.training.utils.layer_control.peft_control_config_parser import LayerControlConfigParser
from cosmos_predict1.diffusion.training.utils.peft.peft import add_lora_layers, setup_lora_requires_grad
from cosmos_predict1.diffusion.utils.customization.customization_manager import CustomizationType
from cosmos_predict1.utils import misc
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_predict1.utils import log

import torch

T = TypeVar("T")

# ★ パラメータ数をカウントするヘルパー関数を追加
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def video_peft_decorator(base_class: Type[T]) -> Type[T]:
    class PEFTVideoDiffusionModel(base_class):
        def __init__(self, config: dict, fsdp_checkpointer=None):
            super().__init__(config)

        @misc.timer("PEFTVideoDiffusionModel: set_up_model")
        def set_up_model(self):
            config = self.config
            peft_control_config_parser = LayerControlConfigParser(config=config.peft_control)
            peft_control_config = peft_control_config_parser.parse()
            self.model = self.build_model()
            if peft_control_config and peft_control_config["customization_type"] == CustomizationType.LORA:
                add_lora_layers(self.model, peft_control_config)

                # ★★★★★ ここからデバッグログ出力コードを追加 ★★★★★
                trainable_before, total_before = count_parameters(self.model)
                print(f"[PEFT DEBUG] Before LoRA setup: Trainable params: {trainable_before:,} / Total params: {total_before:,} ({trainable_before/total_before:.2%})")

                print("[DEBUG] setup_lora_requires_grad is being called.")
                num_lora_params = setup_lora_requires_grad(self.model)

                trainable_after, total_after = count_parameters(self.model)
                print(f"[PEFT DEBUG] After LoRA setup: Trainable params: {trainable_after:,} / Total params: {total_after:,} ({trainable_after/total_after:.2%})")
                print(f"[PEFT DEBUG] LoRA params configured by setup_lora_requires_grad: {num_lora_params:,}")
                # ★★★★★ ここまで ★★★★★

                if num_lora_params == 0:
                    raise ValueError("No LoRA parameters found. Please check the model configuration.")
            else:
                print("[DEBUG] LoRA setup skipped. Check customization_type or peft_control_config.")
            if config.ema.enabled:
                with misc.timer("PEFTDiffusionModel: instantiate ema"):
                    config.ema.model = self.model
                    self.model_ema = lazy_instantiate(config.ema)
                    config.ema.model = None
            else:
                self.model_ema = None

        # 追加
        def clip_grad_norm_(self, max_norm, norm_type=2):
            """
            Performs gradient clipping.
            Internally, it calls the standard PyTorch function.
            """
            # このモデルの学習対象パラメータに対して勾配クリッピングを実行します
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            return torch.nn.utils.clip_grad_norm_(trainable_params, max_norm, norm_type=norm_type)

        def state_dict_model(self) -> Dict:
            return {
                "model": self.model.state_dict(),
                "ema": self.model_ema.state_dict() if self.model_ema else None,
            }

    return PEFTVideoDiffusionModel


@video_peft_decorator
class PEFTVideoDiffusionModel(VideoDiffusionModel):
    pass


@video_peft_decorator
class PEFTExtendDiffusionModel(ExtendDiffusionModel):
    pass
