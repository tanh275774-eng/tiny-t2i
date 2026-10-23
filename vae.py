"""冻结的预训练 VAE。显式展示像素范围与 latent scaling，避免训练/推理不一致。"""

import torch
from torch import nn

DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


class FrozenVAE(nn.Module):
    def __init__(self, model_id=DEFAULT_VAE, revision=None):
        super().__init__()
        from diffusers import AutoencoderKL

        self.model = AutoencoderKL.from_pretrained(model_id, revision=revision)
        self.model.requires_grad_(False)
        self.model.eval()
        self.scale = self.model.config.scaling_factor
        self.downsample_factor = 2 ** (len(self.model.config.block_out_channels) - 1)
        self.latent_channels = self.model.config.latent_channels
        if getattr(self.model.config, "shift_factor", None) not in (None, 0):
            raise ValueError("This teaching wrapper supports SD-style VAEs without latent shifts")

    @torch.no_grad()
    def encode(self, images, sample_posterior=False):
        # 输入 [B, 3, H, W]，像素已经归一化到 [-1, 1]。
        self.model.eval()
        posterior = self.model.encode(images).latent_dist
        # 教学实验默认用后验均值，便于缓存和重建对比；也可从后验分布采样。
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return latent * self.scale

    @torch.no_grad()
    def decode(self, latent):
        self.model.eval()
        # 编码时乘过 scaling_factor，解码前必须除回去。
        pixels = self.model.decode(latent / self.scale).sample
        return (pixels / 2 + 0.5).clamp(0, 1)
