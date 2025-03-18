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

import copy
import gc
import random
from contextlib import nullcontext
from typing import ContextManager, List, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict1.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos_predict1.autoregressive.configs.inference.inference_config import TrainingSamplingConfig as SamplingConfig
from cosmos_predict1.autoregressive.utils.parallel import broadcast_data_batch_in_tp_cp_group, get_batch_on_this_cp_rank
from cosmos_predict1.utils import log, misc

_I_FRAMES_IN_PROMPT = 1


def generate_video_from_tokens(
    model: torch.nn.Module,
    prompt_tokens: list[torch.Tensor],
    latent_shape: list[int],
    video_start_boundary: int,
    max_gen_len: int,
    sampling_config: SamplingConfig,
    logit_clipping_range: list[int],
    seed: int = 0,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    action: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Function to generate video from input tokens. These input tokens can be initial text tokens (in case of text to video),
    or partial ground truth tokens.

    Args:
        model (torch.nn.Module): Model instance
        prompt_tokens (list): Prompt tokens used by the model
        latent_shape (list): Shape of the video latents
        video_start_boundary (int): Index where the video tokens start
        max_gen_len (int): Maximum length of the tokens that needs to be generated
        sampling_config (SamplingConfig): Config used by sampler during inference
        logit_clipping_range (list): Range of indices in the logits to be clipped, e.g. [video_token_start, video_token_end]
        context (Optional[torch.Tensor]): The context tensor added via cross-attn.
        context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.
        action (Optional[torch.Tensor]): The robot action tensor added via cross-attn.
    Returns:
        video_decoded (torch.Tensor): Generated video, shape [B=1, C=3, L, H, W], range [0, 1]
        generation_tokens (torch.Tensor): Generated tokens, shape [B=1, 1, LHW], dtype torch.LongTensor
    """

    # Sample the output tokens
    total_seq_len = np.prod(latent_shape)
    if sampling_config.fast_generate:
        assert (
            model.config.backend == "pytorch"
        ), f"Fast generate is only supported for PyTorch backend, got {model.parameters['backend']}"
        assert not sampling_config.logprobs
        assert logit_clipping_range == [
            0,
            model.tokenizer.video_vocab_size,
        ], f"logit_clipping_range {logit_clipping_range} is not supported for fast generate. Expected [0, {model.tokenizer.video_vocab_size}]"
        generation_tokens, _ = model.fast_generate(
            prompt_tokens=prompt_tokens,
            temperature=sampling_config.temperature,
            top_k=sampling_config.top_k,
            top_p=sampling_config.top_p,
            echo=sampling_config.echo,
            seed=seed,
            context=context,
            context_mask=context_mask,
            action=action,
            max_gen_len=max_gen_len,
            compile_decode=sampling_config.compile_decode,
            compile_prefill=sampling_config.compile_prefill,
            verbose=True,
        )
        generation_tokens = generation_tokens[:, video_start_boundary:]
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        if generation_tokens.shape[1] < total_seq_len:
            log.warning(
                f"Generated video tokens (shape:{generation_tokens.shape}) shorted than expected {total_seq_len}. Could be the model produce end token early. Repeat the last token to fill the sequence in order for decoding."
            )
            padding_len = total_seq_len - generation_tokens.shape[1]
            padding_tokens = generation_tokens[:, [-1]].repeat(1, padding_len)
            generation_tokens = torch.cat([generation_tokens, padding_tokens], dim=1)
        # Cast to LongTensor
        indices_tensor = generation_tokens.long()
    else:
        generation_tokens, _ = model.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            top_k=sampling_config.top_k,
            logprobs=sampling_config.logprobs,
            echo=sampling_config.echo,
            logit_clipping_range=logit_clipping_range,
            seed=seed,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        generation_tokens = [g[video_start_boundary:] for g in generation_tokens]
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        for i in range(len(generation_tokens)):
            if len(generation_tokens[i]) < total_seq_len:
                log.warning(
                    f"Generated video tokens {len(generation_tokens[i])} shorted than expected {total_seq_len}. Could be the model produce end token early. Repeat the last token to fill the sequence in order for decoding."
                )
                generation_tokens[i] = generation_tokens[i] + [generation_tokens[i][-1]] * (
                    total_seq_len - len(generation_tokens[i])
                )

        indices_tensor = torch.LongTensor(generation_tokens)
    # First, we reshape the generated tokens into batch x time x height x width
    indices_tensor = rearrange(
        indices_tensor,
        "B (T H W) -> B T H W",
        T=latent_shape[0],
        H=latent_shape[1],
        W=latent_shape[2],
    )
    log.info(f"generated video tokens {len(generation_tokens[0])} -> reshape: {indices_tensor.shape}")
    # If logit clipping range is specified, offset the generated indices by the logit_clipping_range[0]
    # Video decoder always takes tokens in the range (0, N-1). So, this offset is needed.
    if len(logit_clipping_range) > 0:
        indices_tensor = indices_tensor - logit_clipping_range[0]

    # Now decode the video using tokenizer.
    if model.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap == 0:
        video_decoded = model.tokenizer.video_tokenizer.decode(indices_tensor.cuda())
    else:
        video_decoded = model.tokenizer.video_tokenizer.decode_with_overlap(
            indices_tensor.cuda(), temporal_overlap=model.tokenizer.tokenizer_config.video_tokenizer.temporal_overlap
        )

    # Normalize decoded video from [-1, 1] to [0, 1], and clip value
    video_decoded = (video_decoded * 0.5 + 0.5).clamp_(0, 1)
    return video_decoded, generation_tokens, indices_tensor


@torch.inference_mode()
def generate_parital_tokens_from_data_batch(
    model: torch.nn.Module,
    data_batch: dict,
    num_tokens_to_generate: int,
    sampling_config: SamplingConfig,
    tokenizer_config: TokenizerConfig,
    latent_shape: list[int],
    task_condition: str,  # Please refer to parser.add_argument("--task_condition",...) in this file for choices
    max_samples: int = None,
    num_chunks_to_generate: int = 1,
    latent_context_t_size: int = 2,
    seed: int = 0,
    action: Optional[torch.Tensor] = None,
) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Generate video from partial tokens. This function is used for sampling with conditioning.
    Args:
        model (torch.nn.Module): Model instance
        data_batch (dict): Data batch
        num_tokens_to_generate (int): Number of tokens to generate
        sampling_config (SamplingConfig): Sampling configuration
        tokenizer_config (TokenizerConfig): Tokenizer configuration, contains video_tokenizer config
        latent_shape (list[int]): Shape of the video latents
        max_samples (int): Maximum number of samples to generate
        latent_context_t_size is always effective for the long video generation, but not always used for the first chunk generation, how many tokens to use for the first token depends on the task_condition, for example, if task_condition is text only, then latent_context_t_size won't be used for 1st chunk
        action (Optional[torch.Tensor]): The robot action tensor added via cross-attn.  None or [B, action_dim]
    Returns:
        out_videos (List[torch.Tensor]): List of generated videos, each entry has shape [C, L, H, W], range [0, 1]
        inp_videos (List[torch.Tensor]): List of input videos, each entry has shape [C, L, H, W], range [0, 1]
        out_generated_tokens (List[torch.Tensor]): List of generated tokens, each entry is of shape [B=1, 1, LHW], dtype torch.LongTensor
    """

    log.info(f"Starting generate_parital_tokens_from_data_batch with seed {seed}")
    log.info(f"Number of tokens to generate: {num_tokens_to_generate}")
    log.info(f"Latent shape: {latent_shape}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        broadcast_data_batch_in_tp_cp_group(data_batch)

    video_token_start = tokenizer_config.video_tokenizer.tokenizer_offset
    video_vocab_size = tokenizer_config.video_tokenizer.vocab_size
    video_token_end = video_token_start + video_vocab_size

    logit_clipping_range = [video_token_start, video_token_end]

    out_videos = {}
    inp_videos = []
    out_generated_tokens = []
    out_indices_tensors = {}
    ignore_num_t = int(
        _I_FRAMES_IN_PROMPT
        + (latent_context_t_size - _I_FRAMES_IN_PROMPT) * tokenizer_config.video_tokenizer.config.compression_ratio[0]
    )
    # for text2video, we only add a <bov> token at the beginning of the video tokens
    if model.tokenizer.tokenizer_config.training_type == "text_to_video":
        num_bov_tokens = 1
        num_eov_tokens = 0
    else:
        num_eov_tokens = 1 if model.tokenizer.tokenizer_config.add_special_tokens else 0
        num_bov_tokens = 1 if model.tokenizer.tokenizer_config.add_special_tokens else 0

    for chunk_idx in range(num_chunks_to_generate):
        out_videos[chunk_idx] = None
        out_indices_tensors[chunk_idx] = None

        # get the context embedding and mask
        context = data_batch.get("context", None) if task_condition != "video" else None
        context_mask = data_batch.get("context_mask", None) if task_condition != "video" else None
        if context is not None:
            context = misc.to(context, "cuda").detach().clone()
        if context_mask is not None:
            context_mask = misc.to(context_mask, "cuda").detach().clone()

        if (
            task_condition == "text_and_first_random_token" or task_condition == "text_and_first_bov_token"
        ) and chunk_idx == 0:
            # In this case, we don't need to initialize CP and encode the videos,
            # because we don't have video key in the data_batch, we only need to generate the first random token
            batch_size = data_batch["context"].shape[0]
        else:
            # get the video tokens
            data_tokens, token_boundaries = model.tokenizer.tokenize(data_batch=data_batch)
            data_tokens = misc.to(data_tokens, "cuda").detach().clone()
            if parallel_state.get_context_parallel_world_size() > 1:
                data_tokens = get_batch_on_this_cp_rank(data_tokens)
            batch_size = data_tokens.shape[0]

        # Changing this to batched inference
        # for sample_num in range(batch_size):
        if task_condition == "text_and_first_random_token":
            if chunk_idx == 0:
                if model.tokenizer.tokenizer_config.add_special_tokens:
                    raise ValueError("We don't support adding special token yet")
                else:
                    random_tokens = [
                        [random.randint(0, model.tokenizer.tokenizer_config.video_tokenizer.vocab_size - 1)]
                        for _ in range(batch_size)
                    ]
                    input_tokens = random_tokens
                video_start_boundary = num_bov_tokens
            else:
                ValueError("long video for text2video is not verified and tested yet!")
        elif task_condition == "text_and_first_bov_token":
            if chunk_idx == 0:
                input_tokens = [[model.tokenizer.video_special_tokens["<|begin_of_video|>"]] * batch_size]
                video_start_boundary = num_bov_tokens
            else:
                ValueError("long video for text2video is not verified and tested yet!")
        elif task_condition == "text_and_first_gt_token":
            if chunk_idx == 0:
                input_tokens = input_tokens[: (num_bov_tokens + 1)].tolist()
                input_tokens = [
                    input_tokens
                ]  # if num_bov_tokens = 1, we need to take the first two tokens, otherwise only the first one
                video_start_boundary = token_boundaries["video"][sample_num][0] + num_bov_tokens
            else:
                ValueError("long video for text2video is not verified and tested yet!")
        else:
            input_tokens = []
            for sample_num in range(batch_size):
                input_tokens_cur = data_tokens[sample_num][0 : token_boundaries["video"][sample_num][1]]  # [B, L]
                input_tokens.append(input_tokens_cur[0 : -num_tokens_to_generate - num_eov_tokens].tolist())
            log.info(
                f"Run sampling. # input condition tokens: {len(input_tokens[0])}; # generate tokens: {num_tokens_to_generate + num_eov_tokens}; "
                f"full length of the data tokens: {len(data_tokens[sample_num])}: {data_tokens[sample_num]}"
            )
            video_start_boundary = token_boundaries["video"][sample_num][0] + num_bov_tokens

        current_action = action[sample_num : sample_num + 1, :] if task_condition == "video_and_action" else None

        video_decoded, generated_tokens, indices_tensor = generate_video_from_tokens(
            model=model,
            prompt_tokens=input_tokens,
            latent_shape=latent_shape,
            video_start_boundary=video_start_boundary,
            max_gen_len=num_tokens_to_generate,
            sampling_config=sampling_config,
            logit_clipping_range=logit_clipping_range,
            seed=seed,
            context=context,
            context_mask=context_mask,
            action=current_action,
        )  # video_decoded is of shape BCLHW, range [0, 1]
        # indices_tensor is of shape BLHW
        out_generated_tokens.append(generated_tokens)  # output generated tokens

        if chunk_idx == 0:
            if (task_condition not in ["text_and_first_random_token", "text_and_first_bov_token"]) or (
                "video" in data_batch.keys()
            ):
                # For first random token generation, there is not input videos, so we don't need to append the input videos
                inp_videos = data_batch["video"].detach().clone() * 0.5 + 0.5
            # For the first chunk, we store the entire generated video
            out_videos[chunk_idx] = video_decoded.detach().clone()
            out_indices_tensors[chunk_idx] = indices_tensor.detach().clone()

        else:
            # For subsequent chunks, we only store the newly generated part
            # We ignore the first `ignore_num_t` frames as they overlap with the previous chunk
            out_videos[chunk_idx] = video_decoded[:, :, ignore_num_t:, :, :].detach().clone()
            # BCLHW, range [0, 1]
            log.warning(
                "long video for diffusion decoder is not verified and tested yet! So we don't add chunk_idx>0 for out_indices_tensors"
            )

        if task_condition not in ["text_and_first_random_token", "text_and_first_bov_token"]:
            # Update the input for the next chunk
            # We take the last `ignore_num_t` frames of the generated video
            # and use them as the first `ignore_num_t` frames for the next chunk
            data_batch["video"][sample_num, :, :ignore_num_t, :, :] = (
                video_decoded[0, :, -ignore_num_t:, :, :].detach().clone() - 0.5
            ) * 2
        else:
            ValueError("long video for text2video is not verified and tested yet!")

    # output_videos = []
    # output_indice_tensors = []

    tensors_to_concat = [out_videos[chunk_idx] for chunk_idx in range(num_chunks_to_generate)]
    output_videos = torch.cat(tensors_to_concat, dim=2)

    indices_tensor_to_concat = [out_indices_tensors[chunk_idx] for chunk_idx in range(num_chunks_to_generate)]
    output_indice_tensors = torch.cat(indices_tensor_to_concat, dim=1)  # BLHW

    if task_condition in ["text_and_first_random_token", "text_and_first_bov_token"] and len(inp_videos) == 0:
        inp_videos = None

    return output_videos, inp_videos, out_generated_tokens, output_indice_tensors
