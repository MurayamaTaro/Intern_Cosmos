## Post-training autoregressive-based base models

### Model Support Matrix

We support the following Cosmos Autoregressive models for post-training. Review the available models and their compute requirements for post-training and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-Predict1-4B           | **Supported**    | 1 NVIDIA GPUs*                           |

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
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_autoregressive_checkpoints.py --model_sizes 4B
   ```
4. For tensor parallel training, checkpoints need to be sharded to the target tensor model parallel size TP. Shard checkpoints to TP=4 with:
   ```bash
    python scripts/shard_autoregressive_base_checkpoints.py --checkpoint_path checkpoints/Cosmos-Predict1-4B/model.pt --model_size 4b --tensor_parallel_size 4  
   ```

### Examples

Post-training a Cosmos Autoregressive WFM enables you to train the model to generate videos that are more specific to your use case.

There are 3 steps to post-training: downloading a dataset, preprocessing the data, and post-training the model.


#### 1. Download a Dataset
The Bridge dataset can be downloaded from [IRASim](https://github.com/bytedance/IRASim). We just need the raw RGB videos.

#### 2. Post-train the Model with Mock Data

Run the following command to execute an example post-training job with mock data
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=1 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock job.wandb_mode="online"
```

The model will be post-trained using mock data.
See the config `VIDEO2WORLD_FT_4B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK1` defined in `cosmos_predict1/autoregressive/configs/experiment/video2video/basic.py` to understand how the dataloader is determined.

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `autoregressive_base`, `NAME` is `video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock`.

See the job config to understand how they are determined.
```python
VIDEO2WORLD_FT_4B_TP1_CP1_PT_DDP_FRAME36_CHUNK9_MOCK1= LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="autoregressive_base",
            name="video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining_autoregressive_base/video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock/checkpoints/
├── iter_{NUMBER}.pt
```

#### 3. Post-train the Model with Bridge Data

Run the following command to execute an example post-training job with bridge data by training from scratch with TP=4.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge job.wandb_mode="online"
```

We can also load the pre-trained checkpoints and fine-tune it on the bridge data for faster convergence with TP=4.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=4 -m cosmos_predict1.autoregressive.train --config=cosmos_predict1/autoregressive/configs/config.py -- experiment=video2world_ft_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge job.wandb_mode="online"
```

The model will be post-trained using bridge data.
See the `BridgeDataset` defined in `cosmos_predict1/autoregressive/datasets/bridge_dataset.py` to understand how the dataloader is determined.

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`. Taking the fine-tuning job as an example:
`PROJECT` is `posttraining`, `GROUP` is `autoregressive_base`, `NAME` is `video2world_ft_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge`.

See the job config to understand how they are determined.
```python
VIDEO2WORLD_FT_4B_TP4_CP1_PT_DDP_FRAME33_CHUNK33_BRIDGE= LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="autoregressive_base",
            name="video2world_ft_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining_autoregressive_base/video2world_ft_4b_tp4_cp1_pt_ddp_frame33_chunk33_bridge/checkpoints/
├── iter_{NUMBER}.pt
```

### Inference with the Post-trained Model Checkpoint

The inference can be done with the same interface as described in [examples/inference_autoregressive_base.md](examples/inference_autoregressive_base.md).

#### 1. Copying checkpoint to Designated Location

The post-trained checkpoint needs to be copied to `checkpoints/Cosmos-Predict1-4B-Base_post-trained/model.pt`

For example, with TP=1 if a posttrained checkpoint with 1000 iterations is to be used,
```bash
# copy checkpoint to the designated location
mkdir checkpoints/Cosmos-Predict1-4B-Base_post-trained_post-trained/
cp checkpoints/posttraining_autoregressive_base/video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock/checkpoints/iter_000001000.pt checkpoints/Cosmos-Predict1-4B-Base_post-trained/model.pt
```

With TP=4, the postrained checkpoints are sharded and should first be merged into a single checkpoint for inference
```bash
python scripts/merge_autoregressive_tp_checkpoints.py --checkpoint_path checkpoints/posttraining_autoregressive_base/video2world_ft_4b_tp1_cp1_pt_ddp_frame36_chunk9_mock/checkpoints/iter_000001000.pt --output_path checkpoints/Cosmos-Predict1-4B-Base_post-trained/model.pt --model_size 4b --tensor_parallel_size 4
```

#### 2. Running the Inference

This is the basic example for running inference on the TP=1 post-trained 4B model with a single video.

```bash
NUM_GPUS=<NUM_GPUS>
PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/base.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B-Base_post-trained \
    --input_type video \
    --input_image_or_video_path assets/autoregressive/input.mp4 \
    --top_p 0.8 \
    --temperature 1.0 \
    --offload_diffusion_decoder \
    --offload_tokenizer \
    --video_save_name autoregressive-4b-post-train
```
