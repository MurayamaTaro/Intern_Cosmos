#!/usr/bin/env python3
"""
Unified script: merge LoRA adapter into base model and perform batch inference
for Cosmos-Predict1 Text2World before/after comparison.
"""
import subprocess
import time
from pathlib import Path
import torch
from peft import PeftModel

# === Configuration ===
NUM_GPUS = 8
CHECKPOINT_DIR = Path("checkpoints")

# Model directory names under CHECKPOINT_DIR
BASE_MODEL_NAME = "Cosmos-Predict1-7B-Text2World"
LORA_MODEL_NAME = "Cosmos-Predict1-7B-Text2World_post-trained-lora"
MERGED_MODEL_NAME = "Cosmos-Predict1-7B-Text2World-LORA-merged"

# Paths
BASE_MODEL_DIR = CHECKPOINT_DIR / BASE_MODEL_NAME
LORA_MODEL_DIR = CHECKPOINT_DIR / LORA_MODEL_NAME
MERGED_MODEL_DIR = CHECKPOINT_DIR / MERGED_MODEL_NAME

OUT_ROOT = Path("outputs/comparison")
NUM_STEPS = 35
SEED = 0
FPS = 24

# Prompts for comparison
PROMPTS = {
    "prompt1_pick_and_place": "A video of sks teal robot picking up a green cube and placing it on a red platform.",
    "prompt2_rotate": "A video of sks teal robot rotating 360 degrees in front of a yellow background.",
    "prompt3_push_balls": "A video of sks teal robot pushing two blue balls across a table.",
}

def merge_lora():
    """Merge LoRA adapter into base model and save to MERGED_MODEL_DIR/model.pt"""
    MERGED_MODEL_DIR.mkdir(exist_ok=True)
    merged_file = MERGED_MODEL_DIR / "model.pt"
    if merged_file.exists():
        print("LoRA merged model already exists, skipping merge.")
        return

    print("Merging LoRA adapter into base model...")
    # Load base state dict
    base_state = torch.load(BASE_MODEL_DIR / "model.pt", map_location="cpu")
    # Wrap and merge
    peft_model = PeftModel.from_pretrained(base_state, LORA_MODEL_DIR)
    merged_model = peft_model.merge_and_unload()
    # Save merged state_dict
    torch.save(merged_model.state_dict(), merged_file)
    print(f"Merged model saved to {merged_file}")


def run_inference(model_name: str, prompt: str, output_dir: Path, name: str, disable_upsampler: bool) -> float:
    """Run torchrun inference and return elapsed seconds."""
    cmd = [
        "torchrun", f"--nproc_per_node={NUM_GPUS}",
        "cosmos_predict1/diffusion/inference/text2world.py",
        f"--num_gpus={NUM_GPUS}",
        f"--checkpoint_dir={CHECKPOINT_DIR}",
        f"--diffusion_transformer_dir={model_name}",
        f"--prompt={prompt}",
        f"--num_steps={NUM_STEPS}",
        f"--video_save_folder={output_dir}",
        f"--video_save_name={name}",
        f"--seed={SEED}",
        f"--fps={FPS}",
        "--disable_guardrail",
    ]
    if disable_upsampler:
        cmd.append("--disable_prompt_upsampler")

    start = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - start


def main():
    # Prepare output
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Merge LoRA once
    merge_lora()

    # Inference configurations
    settings = {
        "before": {"model": BASE_MODEL_NAME, "disable_upsampler": False},
        "after": {"model": MERGED_MODEL_NAME, "disable_upsampler": True},
    }

    # Batch process each prompt
    for key, orig_prompt in PROMPTS.items():
        for phase, cfg in settings.items():
            out_dir = OUT_ROOT / key / phase
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n➡️ Running '{phase}' for '{key}'")
            elapsed = run_inference(
                cfg["model"], orig_prompt if phase=="before" else (out_dir / f"{key}_before.txt").read_text().strip(),
                out_dir, f"{key}_{phase}", cfg["disable_upsampler"]
            )
            print(f"⏱ {phase.capitalize()} elapsed: {elapsed:.2f} s")

    print("\n🎉 All done! Check videos in 'outputs/comparison'.")

if __name__ == "__main__":
    main()
