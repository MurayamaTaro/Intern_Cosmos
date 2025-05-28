import torch, re
sd = torch.load("checkpoints/Cosmos-Predict1-7B-Text2World_post-trained-lora/model.pt", map_location="cpu")["model"]

print(list(sd.keys()))

w = sd["net.blocks.block0.blocks.0.block.attn.to_q.0.weight"]
print("mean", w.mean().item(), "std", w.std().item())
