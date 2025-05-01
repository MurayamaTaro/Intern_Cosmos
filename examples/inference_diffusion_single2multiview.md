## Inference with diffusion-based Single2Multiview models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Sample-AV-Single2Multiview --checkpoint_dir checkpoints
   ```

### Examples
We provide `Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Text2World` for world generation from front view video and text, 
and `Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Video2World` for world generation from front view video and multiview initial frame(s).

The inference script is `cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py`.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py --help
```

We will set the prompt with an environment variable first.
```bash
PROMPT="The video is captured from a camera mounted on a car. The camera is facing forward. \
The video is taken from the perspective of a vehicle's dashboard camera, showing a straight road flanked by snow-covered trees and a clear sky. \
The road is mostly empty, with no visible traffic or pedestrians. \
The sun is setting, casting a warm glow on the horizon and creating long shadows on the snow. \
The trees are tall and leafless, with some coniferous trees interspersed among the bare deciduous trees. \
The snow on the ground appears undisturbed, suggesting a quiet and peaceful setting."

PROMPT_LEFT="The video is captured from a camera mounted on a car. The camera is facing to the left. \
The video captures a series of images from a moving vehicle, showcasing a winter scene with snow-covered ground and trees. \
The sky is a gradient of blue and orange hues, indicating either sunrise or sunset. \
The trees are tall and predominantly coniferous, with some deciduous trees as well. \
The snow appears undisturbed, suggesting a quiet, possibly early morning setting. \
There are no visible people or animals, and the road is clear of traffic. \
The video has a fisheye lens effect, which gives a wide-angle view of the surroundings."

PROMPT_RIGHT="The video is captured from a camera mounted on a car. The camera is facing to the right. \
The video captures a series of images taken from a moving vehicle, showcasing a winter scene with snow-covered ground and trees. \
The sky is a gradient of blue hues, indicating either dawn or dusk. \
The trees are predominantly coniferous, with some bare deciduous trees. \
The snow appears fresh and undisturbed, suggesting recent snowfall. \
There are no visible people or animals, and the environment is serene and untouched. \
The perspective changes as the vehicle moves, providing different angles of the same landscape."

PROMPT_BACK="The video is captured from a camera mounted on a car. The camera is facing backwards. \
The video captures a sequence of frames showing a road covered in snow, with tire tracks visible on the surface. \
The road is flanked by tall, leafless trees, and the sky is a gradient of pink and blue hues, indicating either sunrise or sunset. \
The lighting conditions suggest it is either early morning or late evening. \
There are no visible signs of people or animals, and the road appears to be in a rural or less populated area. \
The vehicles in the video are moving at a steady pace, and there are no visible traffic signs or markings that stand out."

PROMPT_BACK_LEFT="The video is captured from a camera mounted on a car. The camera is facing the rear left side."

PROMPT_BACK_RIGHT="The video is captured from a camera mounted on a car. The camera is facing the rear right side."
```

#### Example 1: single view extension with Text condition
This is the basic example for running inference on the 7B single to multiview model with a single view input video. 
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Text2World \
    --view_condition_video assets/diffusion/sv2mv_input1.mp4 \
    --num_input_frames 1 \
    --condition_location "first_cam" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-text2world
```
Similar to other examples, multiple gpus can be leveraged in generation too:
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Text2World \
    --view_condition_video assets/diffusion/sv2mv_input1.mp4 \
    --num_input_frames 1 \
    --condition_location "first_cam" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-text2world-8gpu
```
#### Example 2: single view extension with initial frames condition
This example runs the front view + initial frames extension into multiview video.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Video2World \
    --view_condition_video assets/diffusion/sv2mv_input1.mp4 \
    --initial_condition_video assets/diffusion/sv2mv_init.mp4 \
    --num_input_frames 9 \
    --condition_location "first_cam_and_first_n" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-video2world
```

#### Example 3: single view extension with looped generation
This example uses the generation results of example 1 as initial frames input to the video2world model for long video generation.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_view_extend_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Video2World \
    --view_condition_video assets/diffusion/sv2mv_input1.mp4 \
    --initial_condition_video outputs/diffusion-single2multiview-text2world.mp4 \
    --num_input_frames 9 \
    --view_cond_start_frame 48 \
    --condition_location "first_cam_and_first_n" \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-single2multiview-video2world-lvg
```

