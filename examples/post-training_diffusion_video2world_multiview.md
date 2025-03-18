## Post-training diffusion-based Video2World models (with multi-view data)

### Model Support Matrix

We support the following Cosmos Diffusion models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-7B-Video2World           | **Supported**    | 8 NVIDIA GPUs*                           |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

### Environment setup
Clone the `cosmos-predict1` source code
```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict1.git
cd cosmos-predict1
```

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

```bash
# Create the cosmos conda environment.
conda env create --file cosmos.yaml
# Activate the cosmos conda environment.
conda activate cosmos
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
# Install Apex for full training with bfloat16.
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .
```

You can test the environment setup with
```bash
PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Video2World-Sample-AV-Multiview
   ```

### Examples

#### Post-train the Model (with mock data)

Run the following command to execute an example post-training job with the mock data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train --config=cosmos_predict1/diffusion/training/config/config_multiview.py -- experiment=video2world_multiview_7b_example
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_video2world`, `NAME` is `video2world_multiview_7b_example`.

See the job config to understand how they are determined.
```python
video2world_multiview_7b_example = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_video2world",
            name="video2world_multiview_7b_example",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_video2world/video2world_multiview_7b_example/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_video2world.md](examples/inference_diffusion_video2world.md).

#### 1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview_post-trained/model.pt`

For example, if a post-trained checkpoint (ema) with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview_post-trained/
cp checkpoints/posttraining/diffusion_video2world/video2world_multiview_7b_example/checkpoints/iter_000001000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview_post-trained/model.pt
```
#### 2. Running the Inference

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

```bash
# Run the video generation command with a single gpu
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world_multiview.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview_post-trained \
    --input_image_or_video_path assets/diffusion/video2world_multiview_input1.mp4 \
    --num_input_frames 1 \
    --prompt "${PROMPT}" \
    --prompt_left "${PROMPT_LEFT}" \
    --prompt_right "${PROMPT_RIGHT}" \
    --prompt_back "${PROMPT_BACK}" \
    --prompt_back_left "${PROMPT_BACK_LEFT}" \
    --prompt_back_right "${PROMPT_BACK_RIGHT}" \
    --video_save_name diffusion-video2world-multiview-7b-post-train
```
