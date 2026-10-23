"""从零定义的小型 DiT：2D 位置编码 + 时间调制 + 文本 cross-attention。"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DiTConfig:
    latent_size: int = 16     # 128 像素图片，经 8 倍压缩后得到 16×16 latent
    in_channels: int = 4
    patch_size: int = 2       # latent 的 2×2 patch，不是原图的 2×2 像素
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 8
    text_dim: int = 512


def sinusoidal_embedding(values, dim):
    """给一维坐标/时间生成 sin-cos 特征：[N] -> [N, dim]。"""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=values.device).float() / half
    )
    angles = values.float()[:, None] * frequencies[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def position_embedding_2d(grid_size, dim):
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    # 特征的前后两半分别编码行、列；这是加法位置编码，不是 RoPE。
    return torch.cat([
        sinusoidal_embedding(y.flatten(), dim // 2),
        sinusoidal_embedding(x.flatten(), dim // 2),
    ], dim=-1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context=None, context_mask=None):
        # self-attention: Q/K/V 都来自图像；cross-attention: K/V 来自文本。
        context = x if context is None else context
        batch = x.shape[0]

        def split_heads(value):
            return value.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(self.q(x)), split_heads(self.k(context)), split_heads(self.v(context))
        # SDPA 的 bool mask 中 True 表示允许关注，屏蔽文本 padding。
        mask = None if context_mask is None else context_mask[:, None, None, :].bool()
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, x.shape[1], -1)
        return self.out(attended)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attention = Attention(dim, num_heads)
        self.cross_attention = Attention(dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        # 每个分支有 shift、scale、gate 三组时间条件参数。
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, time_embedding, text, text_mask):
        s1, a1, g1, s2, a2, g2, s3, a3, g3 = self.ada(time_embedding).chunk(9, dim=-1)
        x = x + g1[:, None] * self.self_attention(modulate(self.norm1(x), s1, a1))
        x = x + g2[:, None] * self.cross_attention(
            modulate(self.norm2(x), s2, a2), text, text_mask
        )
        return x + g3[:, None] * self.mlp(modulate(self.norm3(x), s3, a3))


class TinyDiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        c = config
        if min(c.latent_size, c.patch_size, c.hidden_size, c.depth, c.num_heads, c.text_dim, c.in_channels) <= 0:
            raise ValueError("All model dimensions must be positive")
        if c.latent_size % c.patch_size or c.hidden_size % 4 or c.hidden_size % c.num_heads:
            raise ValueError("latent_size must divide by patch_size; hidden_size by 4 and num_heads")
        self.config = c
        self.grid_size = c.latent_size // c.patch_size
        self.patch_embed = nn.Conv2d(c.in_channels, c.hidden_size, c.patch_size, stride=c.patch_size)
        self.register_buffer("pos_embed", position_embedding_2d(self.grid_size, c.hidden_size))
        self.time_mlp = nn.Sequential(nn.Linear(c.hidden_size, c.hidden_size), nn.SiLU(),
                                      nn.Linear(c.hidden_size, c.hidden_size))
        self.text_proj = nn.Linear(c.text_dim, c.hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(c.hidden_size, c.num_heads) for _ in range(c.depth)])
        self.final_norm = nn.LayerNorm(c.hidden_size, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(c.hidden_size, 2 * c.hidden_size))
        self.final_proj = nn.Linear(c.hidden_size, c.patch_size ** 2 * c.in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.reshape(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # adaLN-Zero 风格初始化：起初 block 是恒等映射，速度输出是 0。
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, patches):
        # 每个 token 预测一个 p×p×C 的速度 patch，重排回空间网格。
        c = self.config
        b, g, p = patches.shape[0], self.grid_size, c.patch_size
        patches = patches.reshape(b, g, g, p, p, c.in_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(b, c.in_channels, g * p, g * p)

    def forward(self, latent, t, text, text_mask):
        c = self.config
        if tuple(latent.shape[1:]) != (c.in_channels, c.latent_size, c.latent_size):
            raise ValueError("Latent shape does not match DiTConfig")
        # [B, 4, 16, 16] -> [B, 64, 256]，以下注释以默认配置为例。
        x = self.patch_embed(latent).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(x.dtype)
        time_embedding = self.time_mlp(sinusoidal_embedding(t * 1000, c.hidden_size))
        text = self.text_proj(text)  # [B, 77, 512] -> [B, 77, 256]
        for block in self.blocks:
            x = block(x, time_embedding, text, text_mask)
        shift, scale = self.final_ada(time_embedding).chunk(2, dim=-1)
        return self.unpatchify(self.final_proj(modulate(self.final_norm(x), shift, scale)))
