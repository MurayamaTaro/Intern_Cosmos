import torch, math

base = torch.load(
    "checkpoints/Cosmos-Predict1-7B-Text2World/model.pt",
    map_location="cpu") # ["model"]

lora = torch.load(
    "checkpoints/posttraining/diffusion_text2world/text2world_7b_lora_panda70m_r8_iter3000_bs8_lr0.0001_seed0/" \
    "vehicle/checkpoints/iter_000003000_model.pt", map_location="cpu")["model"]

# ベースと同名キーのパラメータを比較
diff_sum = 0
for k in base.keys():
    if k.endswith("_lora.net.0.weight") or k.endswith("_lora.net.1.weight"):
        continue
    d = (base[k] - lora[k]).float().pow(2).mean().sqrt()
    diff_sum += d.item()
print(f"全体の平均 RMS 差: {diff_sum/len(base):.6f}")
