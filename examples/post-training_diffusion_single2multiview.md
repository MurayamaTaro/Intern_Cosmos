## Post-training diffusion-based Single2Multiview-Sample-AV Models

### Model Support Matrix

We support the following Cosmos Single2Multiview-Sample-AV models for post-training. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                                                | Model Status   | Compute Requirements for Post-Training   |
|-----------------------------------------------------------|----------------|------------------------------------------|
| Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Text2World | **Supported**  | 8 NVIDIA GPUs*                           |
| Cosmos-Predict1-7B-SingleToMultiView-Sample-AV-Video2World| **Supported**  | 8 NVIDIA GPUs*                           |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

### Environment setup

Please refer to the Post-training section of [INSTALL.md](/INSTALL.md#post-training) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Cosmos-Predict1-7B-Single2Multiview-Sample-AV --checkpoint_dir checkpoints
   ```

### Examples

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your use case.

#### 1. Data Preparation
The first step is to download a dataset with videos and captions and then preprocess it to our required format.

Example 1. You can use [Waymo Open Dataset](https://waymo.com/open/) for post-training.
Please follow the [instruction](https://github.com/nv-tlabs/cosmos-av-sample-toolkits/blob/main/docs/processing_waymo_for_predict1.md) in [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) to download and convert the Waymo Open Dataset.

The resulting folder structure should look like this:
```
<DATA_ROOT>/waymo/
├── cache/
│   ├── prefix_t5_embeddings_pinhole_front.pkl
│   ├── prefix_t5_embeddings_pinhole_front_left.pkl
│   ├── prefix_t5_embeddings_pinhole_front_right.pkl
│   ├── prefix_t5_embeddings_pinhole_side_left.pkl
│   └── prefix_t5_embeddings_pinhole_side_right.pkl
├── videos/
│   ├── pinhole_front
│       ├── *.mp4
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   ├── pinhole_side_right
│   ...
└── t5_xxl/
    ├── pinhole_front
        └── *.pkl
```

If you've used the multiview caption embedding option above, set `load_mv_emb` to True in `cosmos_predict1/diffusion/training/config/text2world_singletomultiview/experiment.py`
#### 2. Post-train the Model

Run the following command to execute an example post-training job with the above data.
```bash
export OUTPUT_ROOT=checkpoints # default value
torchrun --nproc_per_node=8 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config_multiview.py \
    -- experiment=text2world_singletomultiview_7b_example_waymo
```

Optionally, multi-node training can be done with
```bash
# 4-node training example.
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
    -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config_multiview.py \
    -- experiment=text2world_singletomultiview_7b_example_waymo
```

The model will be post-trained using the above waymo dataset.
See the config `text2world_singletomultiview_7b_example_waymo` defined in `cosmos_predict1/diffusion/training/config/text2world_singletomultiview/experiment.py` to understand how the dataloader is determined.
```python
train_num_views = 3
num_frames = 57
view_keys = ["pinhole_front_left", "pinhole_front", "pinhole_front_right", "pinhole_side_left", "pinhole_side_right"]
example_multiview_dataset_waymo = L(Dataset)(
    dataset_dir="cosmos-av-sample-toolkits/waymo_apr25",
    sequence_interval=1,
    num_frames=num_frames,
    view_keys=view_keys,
    video_size=(576, 1024),
    sample_n_views=train_num_views,
    caption_view_idx_map={0:0,1:1,2:2,3:4,4:5},
)

```

The checkpoints will be saved to `${OUTPUT_ROOT}/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `diffusion_text2world`, `NAME` is `text2world_singletomultiview_7b_example_waymo`.

See the job config to understand how they are determined.
```python
text2world_singletomultiview_7b_example_waymo = LazyDict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="diffusion_text2world",
            name="text2world_singletomultiview_7b_example_waymo",
        ),
        ...
    )
)
```

During the training, the checkpoints will be saved in the below structure.
```
checkpoints/posttraining/diffusion_text2world/text2world_singletomultiview_7b_example_waymo/checkpoints/
├── iter_{NUMBER}_reg_model.pt
├── iter_{NUMBER}_ema_model.pt
```
