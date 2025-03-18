## Post-training diffusion-based Video2World models (with instruction following)

### Environment setup
Clone the `cosmos-predict1` source code
```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict1.git
cd cosmos-predict1
```

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

```bash
# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
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
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Video2World
   ```

### Examples

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your use case.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the model.

#### 1. Download a Dataset

The first step is to download a dataset with videos and captions.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

For example, you can use bridge dataset for post-training.


```bash
# Download metadata with video urls and captions
mkdir -p datasets/bridge
cd datasets/bridge
wget https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz

# Unpack and clean up structure.
tar -xf bridge_train_data.tar.gz
mv opensource_robotdata/bridge/* .
rm -rf opensource_robotdata/
```


#### 2. Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video captions used for post-training:

```bash
# The script will read the captions, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_bridge.py --dataset_path datasets/bridge
```

Dataset folder format:
```
datasets/bridge/
├── annotation/
│   ├── train/
│   │   ├── *.json
│   │   ├── *.pickle
│   ├── val/
│   │   ├── *.json
│   │   ├── *.pickle
├── videos/
│   ├── train/
│   │   ├── 0/rgb.mp4
│   ├── val/
│   │   ├── 0/rgb.mp4
```


#### 3. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train --config=cosmos_predict1/diffusion/training/config/config.py -- experiment=video2world_instruction_bridge_57frames
```

The model will be post-trained using the above hdvila dataset.
See the config `video2world_instruction_bridge_57frames` defined in `cosmos_predict1/diffusion/training/config/video2world_instruction/experiment.py` to understand how the dataloader is determined.

```python
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=57,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    load_action=False,
    load_t5_embeddings=True,
)

dataloader_train = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)
...

video2world_instruction_bridge_57frames = LazyDict(
    dict(
        ...
        dataloader_train=dataloader_train,
        ...
    )
)
...
```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_video2world_instruction`, `NAME` is `video2world_instruction_bridge_57frames`.

See the job config to understand how they are determined.
```python
video2world_instruction_bridge_57frames = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_video2world_instruction",
            name="video2world_instruction_bridge_57frames",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_video2world_instruction/video2world_instruction_bridge_57frames/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```


### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_diffusion_video2world.md](examples/inference_diffusion_video2world.md).

#### 1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-7B-Video2World_instruction_post-trained/model.pt`

For example, if a posttrained checkpoint (ema) with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-7B-Video2World_instruction_post-trained/
cp checkpoints/posttraining/diffusion_video2world_instruction/video2world_instruction_bridge_57frames/checkpoints/iter_000001000_ema_model.pt checkpoints/Cosmos-Predict1-7B-Video2World_instruction_post-trained/model.pt
```
#### 2. Running the Inference

<!-- This is the basic example for running inference on the post-trained 7B model with a single image.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World_instruction_post-trained \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-7b-post-trained
``` -->
