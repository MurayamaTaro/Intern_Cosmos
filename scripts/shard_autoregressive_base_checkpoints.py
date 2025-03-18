import argparse
import torch
from cosmos_predict1.utils import log

from cosmos_predict1.autoregressive.utils.checkpoint import maybe_convert_checkpoint_to_backend, obtain_tensor_parallel_state_dict
from cosmos_predict1.autoregressive.configs.base.model_config import create_video2world_model_config

def shard_checkpoint(checkpoint_path, tensor_parallel_size, model_size, model_family, target_backend="pytorch"):
    assert checkpoint_path.endswith(".pt"), "Checkpoint path must end with .pt"
    assert model_family == "cosmos", "Only cosmos model family is currently supported"
    assert model_size == "4b", "Only 4B model size is currently supported"
    model_config, _ = create_video2world_model_config(model_ckpt_path=checkpoint_path, model_family=model_family, model_size=model_size, tensor_model_parallel_size=tensor_parallel_size, tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit")
    log.info(f"Sharding checkpoint {checkpoint_path} with {tensor_parallel_size} ranks")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    for tensor_parallel_rank in range(tensor_parallel_size):
        shard = obtain_tensor_parallel_state_dict(checkpoint, tensor_parallel_size, tensor_parallel_rank, model_config, target_backend=target_backend)
        shard_path = checkpoint_path.replace(".pt", f"_model_mp_{tensor_parallel_rank}.pt")
        log.info(f"Saving shard {shard_path}")
        torch.save(shard, shard_path)
        
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Shard NVIDIA Cosmos Predict1 autoregressive models"
    )
    parser.add_argument("--checkpoint_path", "-c", type=str, required=True, default="checkpoints/Cosmos-Predict1-4B/model.pt", help="Path to the checkpoint to shard")
    parser.add_argument("--tensor_parallel_size", "-t", type=int, required=True, help="Number of tensor parallel ranks")
    parser.add_argument("--target_backend", "-b", type=str, required=False, default="pytorch", help="Target backend")
    parser.add_argument("--model_size", "-s", type=str, required=True, help="Model size")
    parser.add_argument("--model_family", "-f", type=str, required=False, default="cosmos", help="Model family")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    shard_checkpoint(args.checkpoint_path, args.tensor_parallel_size, args.model_size, args.model_family, args.target_backend)
