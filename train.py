"""训练入口：缓存冻结编码器输出 -> 从零训练 DiT -> 保存 checkpoint。"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from clip import DEFAULT_CLIP, FrozenCLIP
from dataset import ShapesDataset, save_grid
from dit import DiTConfig, TinyDiT
from flow import flow_training_pair
from vae import DEFAULT_VAE, FrozenVAE


@torch.no_grad()
def cache_images(dataset, vae, batch_size, device):
    """小数据集一次编码后缓存在 CPU 内存，训练时无需反复运行 VAE。"""
    latents, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_index, batch in enumerate(loader):
        latent = vae.encode(batch["image"].to(device))
        latents.append(latent.cpu())
        labels.append(batch["label"])
        if batch_index % 20 == 0 or batch_index == len(loader) - 1:
            print(f"Caching {dataset.split}: {batch_index + 1}/{len(loader)}", flush=True)
    return torch.cat(latents), torch.cat(labels)


@torch.no_grad()
def validation_loss(model, latents, labels, text_bank, mask_bank, batch_size, device):
    """固定验证噪声和时间，衡量未见样本上的条件速度 MSE；不是图片质量分数。"""
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(2026)
    total_loss = 0.0
    for start in range(0, len(latents), batch_size):
        clean = latents[start:start + batch_size].to(device)
        indices = labels[start:start + batch_size].to(device)
        noise = torch.randn(clean.shape, generator=generator).to(device)
        t = torch.randn(clean.shape[0], generator=generator).sigmoid().to(device)
        mixed, target = flow_training_pair(clean, noise, t)
        predicted = model(mixed, t, text_bank[indices], mask_bank[indices])
        total_loss += F.mse_loss(predicted.float(), target.float()).item() * len(clean)
    model.train()
    return total_loss / len(latents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/shapes")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--cond-drop", type=float, default=0.1)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP)
    parser.add_argument("--vae-model", default=DEFAULT_VAE)
    parser.add_argument("--clip-revision", default=None)
    parser.add_argument("--vae-revision", default=None)
    args = parser.parse_args()
    positive = (args.samples, args.val_samples, args.batch_size, args.encode_batch_size,
                args.steps, args.save_every, args.log_every, args.image_size,
                args.hidden_size, args.depth, args.heads, args.patch_size)
    if min(positive) <= 0 or args.lr <= 0 or not 0 <= args.cond_drop < 1:
        parser.error("Counts/lr must be positive, and cond-drop must be in [0, 1)")
    if args.image_size < 32 or args.hidden_size % 4 or args.hidden_size % args.heads:
        parser.error("image-size >= 32; hidden-size must divide by 4 and heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use a GPU server or explicitly select --device cpu")
    if args.precision == "bf16" and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("This example enables bf16 only on supported CUDA GPUs")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory; this minimal trainer does not resume old runs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_data = ShapesDataset(args.samples, args.image_size, args.seed, "train")
    val_data = ShapesDataset(args.val_samples, args.image_size, args.seed, "val")

    # 1. 冻结 VAE，把训练/验证图像变成 latent；此阶段还没有训练 DiT。
    vae = FrozenVAE(args.vae_model, args.vae_revision).to(device)
    if args.image_size % (vae.downsample_factor * args.patch_size):
        parser.error("image-size must be divisible by VAE compression × patch-size")
    latent_size = args.image_size // vae.downsample_factor
    latent_channels = vae.latent_channels
    train_latents, train_labels = cache_images(train_data, vae, args.encode_batch_size, device)
    val_latents, val_labels = cache_images(val_data, vae, args.encode_batch_size, device)
    count = min(8, len(val_data))
    originals = torch.stack([val_data[i]["image"] / 2 + 0.5 for i in range(count)])
    reconstructed = vae.decode(val_latents[:count].to(device)).cpu()
    # 相邻两张是原图和重建，方便区分 VAE 损失与 DiT 生成问题。
    save_grid(torch.stack([originals, reconstructed], dim=1).flatten(0, 1), output / "vae-reconstruction.png")
    del vae

    # 2. 只有 108 种 caption，加一个空提示词。一次编码后缓存 token 特征。
    encoder = FrozenCLIP(args.clip_model, args.clip_revision).to(device)
    text_dim = encoder.hidden_size
    captions = train_data.caption_bank + [""]
    null_index = len(captions) - 1
    features, masks = [], []
    for start in range(0, len(captions), 32):
        text, mask = encoder(captions[start:start + 32])
        features.append(text)
        masks.append(mask)
    text_bank, mask_bank = torch.cat(features), torch.cat(masks)
    del encoder, features, masks, text, mask
    if device.type == "cuda":
        torch.cuda.empty_cache()  # 释放已删除编码器留下的 allocator 缓存，不影响其他进程。

    # 3. 这才是要从零训练的网络；优化器里只放 DiT 参数。
    config = DiTConfig(latent_size=latent_size, in_channels=latent_channels,
                       patch_size=args.patch_size, hidden_size=args.hidden_size,
                       depth=args.depth, num_heads=args.heads, text_dim=text_dim)
    model = TinyDiT(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(TensorDataset(train_latents, train_labels), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader)
    print(f"Trainable DiT parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Latents: {tuple(train_latents.shape)}; text bank: {tuple(text_bank.shape)}", flush=True)

    # 4. Rectified Flow / Flow Matching 训练循环。
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        try:
            clean, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            clean, labels = next(iterator)
        clean, labels = clean.to(device), labels.to(device)
        noise = torch.randn_like(clean)
        t = torch.randn(len(clean), device=device).sigmoid()  # logit-normal 时间采样
        noisy_latent, target_velocity = flow_training_pair(clean, noise, t)

        # 一部分样本使用空文本，让模型同时学习无条件预测，推理时才能做 CFG。
        drop = torch.rand(len(clean), device=device) < args.cond_drop
        condition_indices = labels.masked_fill(drop, null_index)
        text, text_mask = text_bank[condition_indices], mask_bank[condition_indices]

        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()
        with context:
            predicted_velocity = model(noisy_latent, t, text, text_mask)
            loss = F.mse_loss(predicted_velocity.float(), target_velocity.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        running_loss += loss.item()

        if step % args.log_every == 0:
            print(f"step {step}/{args.steps}  train_loss={running_loss / args.log_every:.6f}", flush=True)
            running_loss = 0.0
        if step % args.save_every == 0 or step == args.steps:
            val_loss = validation_loss(model, val_latents, val_labels, text_bank, mask_bank,
                                       args.batch_size, device)
            # 教学版本仅保存推理所需状态，不保存 optimizer/RNG；不支持断点续训。
            checkpoint = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "dit_config": asdict(config), "train_args": vars(args),
                          "step": step, "val_loss": val_loss}
            temporary = output / "checkpoint.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "last.pt")
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "train_loss": loss.item(), "val_loss": val_loss}) + "\n")
            print(f"Saved step {step}; validation velocity MSE={val_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
