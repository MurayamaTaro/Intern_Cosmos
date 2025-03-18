<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Website](https://www.nvidia.com/en-us/ai/cosmos/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos-predict1)

[NVIDIA Cosmos](https://www.nvidia.com/cosmos/) is a developer-first world foundation model platform designed to help Physical AI developers build their Physical AI systems better and faster. Cosmos contains

1. Pre-trained models (available via Hugging Face) under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) that allows commercial use of the models for free.
2. Training scripts under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) for post-training the models for various downstream Physical AI applications.

<!-- ------------------------------ -->

## Key Features

Cosmos-Predict1 includes the following features:

- **Diffusion-based world foundation models** for Text2World and Video2World generation, where a user can generate visual simulation based on text prompts and video prompts.
- **Autoregressive-based world foundation models** for Video2World generation, where a user can generate visual simulation based on video prompts and optional text prompts.
- **Image and video tokenizers** for tokenizing videos into continuous tokens (latent vectors) and discrete tokens (integers) efficiently and effectively.

<!-- ------------------------------ -->

## Examples

Inference with pre-trained models:
* [Inference with diffusion-based Text2World models](examples/inference_diffusion_text2world.md)
* [Inference with diffusion-based Video2World models](examples/inference_diffusion_video2world.md)
* [Inference with autoregressive-based base models](examples/inference_autoregressive_base.md)
* [Inference with autoregressive-based Video2World models](examples/inference_autoregressive_video2world.md)
* [Inference with tokenizer models](examples/inference_tokenizer.md)

Post-training models:
* [Post-training diffusion-based Text2World models](examples/post-training_diffusion_text2world.md)
* [Post-training diffusion-based Text2World models (with multi-view data)](examples/post-training_diffusion_text2world_multiview.md)
* [Post-training diffusion-based Video2World models](examples/post-training_diffusion_video2world.md)
* [Post-training diffusion-based Video2World models (with multi-view data)](examples/post-training_diffusion_video2world_multiview.md)
* [Post-training diffusion-based Video2World models (with instruction following)](examples/post-training_diffusion_video2world_instruction.md)
* [Post-training diffusion-based Video2World models (with action control)](examples/post-training_diffusion_video2world_action.md)
* [Post-training autoregressive-based base models](examples/post-training_autoregressive_base.md)
* [Post-training tokenizer models](examples/post-training_tokenizer.md)

Inference with post-trained models:
* [Inference with diffusion-based Text2World models (with multi-view data)](examples/inference_diffusion_text2world_multiview.md)
* [Inference with diffusion-based Video2World models (with multi-view data)](examples/inference_diffusion_video2world_multiview.md)

<!-- ------------------------------ -->

## Model Family

We provide a series of pre-trained models of different families, available for download on Hugging Face.

**Diffusion models**

* [Cosmos-Predict1-7B-Text2World](https://huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World): Text to visual world generation
* [Cosmos-Predict1-14B-Text2World](https://huggingface.co/nvidia/Cosmos-Predict1-14B-Text2World): Text to visual world generation
* [Cosmos-Predict1-7B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-7B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict1-14B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-14B-Video2World): Video + Text based future visual world generation

**Autoregressive models**

* [Cosmos-Predict1-4B](https://huggingface.co/nvidia/Cosmos-Predict1-4B): Future visual world generation
* [Cosmos-Predict1-12B](https://huggingface.co/nvidia/Cosmos-Predict1-12B): Future visual world generation
* [Cosmos-Predict1-5B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-5B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict1-13B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict1-13B-Video2World): Video + Text based future visual world generation

**Tokenizers**

* [Cosmos-Tokenize1-CV8×8×8-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p): Continuous Video Tokenizer with 8x8x8 spatio-temporal compression with, 121 frames context
* [Cosmos-Tokenize1-DV8×16×16-720p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DV8x16x16-720p): Discrete Video Tokenizer with 8x16x16 spatio-temporal compression, and 49 frames context
* [Cosmos-Tokenize1-CI8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CI8x8-360p): Continuous Image Tokenizer with 8x8 spatial compression with low-resolution support
* [Cosmos-Tokenize1-CI16x16-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CI16x16-360p): Continuous Image Tokenizer with 16x16 spatial compression with low-resolution support
* [Cosmos-Tokenize1-CV4×8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-CV4x8x8-360p): Continuous Video Tokenizer with 4x8x8 spatio-temporal compression with low-resolution support
* [Cosmos-Tokenize1-DI8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DI8x8-360p): Discrete Image Tokenizer with 8x8 spatial compression with low-resolution support
* [Cosmos-Tokenize1-DI16x16-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DI16x16-360p): Discrete Image Tokenizer with 16x16 spatial compression with low-resolution support
* [Cosmos-Tokenize1-DV4×8×8-360p](https://huggingface.co/nvidia/Cosmos-Tokenize1-DV4x8x8-360p): Discrete Video Tokenizer with 4x8x8 spatio-temporal compression with low-resolution support

<!-- ------------------------------ -->

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.  
NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).  
NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
