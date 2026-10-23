"""可复现的图文配对数据：颜色 × 形状 × 大小 × 左中右位置。无需下载数据集。"""

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

COLORS = {"red": (220, 45, 45), "green": (40, 170, 70), "blue": (45, 90, 220),
          "yellow": (235, 200, 30), "purple": (155, 60, 200), "orange": (240, 130, 30)}
SHAPES = ("circle", "square", "triangle")
POSITIONS = {"on the left": 0.25, "in the center": 0.50, "on the right": 0.75}
SIZES = {"small": 0.12, "large": 0.19}
COMBINATIONS = list(itertools.product(COLORS, SHAPES, SIZES, POSITIONS))


def caption_for(attributes):
    color, shape, size, position = attributes
    return f"a {size} {color} {shape} {position} on a white background"


class ShapesDataset(Dataset):
    def __init__(self, count=4096, image_size=128, seed=42, split="train"):
        if count <= 0 or image_size < 32 or split not in ("train", "val"):
            raise ValueError("Invalid dataset size, resolution, or split")
        self.count, self.image_size, self.seed, self.split = count, image_size, seed, split
        self.caption_bank = [caption_for(attributes) for attributes in COMBINATIONS]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        # 字符串 seed 使 train/val 使用不同的随机序列，同一个样本可重复生成。
        rng = random.Random(f"{self.seed}/{self.split}/{index}")
        label = rng.randrange(len(COMBINATIONS))
        color, shape, size, position = COMBINATIONS[label]
        scale = self.image_size * 4  # 先画大图，再缩小做抗锯齿
        image = Image.new("RGB", (scale, scale), "white")
        draw = ImageDraw.Draw(image)
        x = (POSITIONS[position] + rng.uniform(-0.025, 0.025)) * scale
        y = (0.5 + rng.uniform(-0.04, 0.04)) * scale
        radius = SIZES[size] * rng.uniform(0.90, 1.10) * scale
        box = (x - radius, y - radius, x + radius, y + radius)
        if shape == "circle":
            draw.ellipse(box, fill=COLORS[color])
        elif shape == "square":
            draw.rectangle(box, fill=COLORS[color])
        else:
            draw.polygon([(x, y - radius), (x - radius, y + radius),
                          (x + radius, y + radius)], fill=COLORS[color])
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        return {"image": pixels / 127.5 - 1, "caption": self.caption_bank[label], "label": label}


def save_grid(pixels, output, columns=4):
    """pixels: [B,3,H,W] 且范围 [0,1]。"""
    pixels = pixels.detach().float().cpu().clamp(0, 1)
    count, _, height, width = pixels.shape
    grid = Image.new("RGB", (columns * width, ((count + columns - 1) // columns) * height), "white")
    for i, tensor in enumerate(pixels):
        array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        grid.paste(Image.fromarray(array), ((i % columns) * width, (i // columns) * height))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/data-preview.png")
    args = parser.parse_args()
    dataset = ShapesDataset(count=16)
    items = [dataset[i] for i in range(16)]
    save_grid(torch.stack([item["image"] / 2 + 0.5 for item in items]), args.output)
    Path(args.output).with_suffix(".json").write_text(
        json.dumps([item["caption"] for item in items], indent=2), encoding="utf-8")
    print("Saved:", args.output)
