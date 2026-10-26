"""图文配对数据：读取以提示词命名的图片，或生成教学用彩色形状。"""

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

COLORS = {"red": (220, 45, 45), "green": (40, 170, 70), "blue": (45, 90, 220),
          "yellow": (235, 200, 30), "purple": (155, 60, 200), "orange": (240, 130, 30)}
SHAPES = ("circle", "square", "triangle")
POSITIONS = {"on the left": 0.25, "in the center": 0.50, "on the right": 0.75}
SIZES = {"small": 0.12, "large": 0.19}
COMBINATIONS = list(itertools.product(COLORS, SHAPES, SIZES, POSITIONS))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_images(root):
    """递归扫描图片；忽略隐藏文件/文件夹，返回稳定排序的路径。"""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths, seen = [], set()
    for path in sorted(root.rglob("*")):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # 同一个文件的符号链接不重复收录，防止进入两个 split。
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.stem.strip():
            raise ValueError(f"Image filename must contain a caption: {path}")
        seen.add(resolved)
        paths.append(path)
    if not paths:
        raise ValueError(f"No supported images found in: {root}")
    return paths


class ImageCaptionDataset(Dataset):
    """文件名去掉最后一个扩展名即 caption；label 仅用于查找文本缓存。"""

    def __init__(self, paths, image_size=128, caption_bank=None, split="all"):
        if not paths or image_size < 32:
            raise ValueError("Need images and image_size >= 32")
        self.paths = [Path(path) for path in paths]
        self.image_size, self.split = image_size, split
        self.captions = [path.stem.strip() for path in self.paths]
        if any(not caption for caption in self.captions):
            raise ValueError("Image filenames must contain nonempty captions")
        self.caption_bank = (list(caption_bank) if caption_bank is not None
                             else list(dict.fromkeys(self.captions)))
        caption_to_label = {caption: i for i, caption in enumerate(self.caption_bank)}
        self.labels = [caption_to_label[caption] for caption in self.captions]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                # 透明区域合成到白底；灰度图等也统一变成 RGB 三通道。
                background = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(background, image).convert("RGB")
                # 保留完整画面，不拉伸、不裁掉主体、不做可能改变 caption 的翻转。
                image = ImageOps.pad(image, (self.image_size, self.image_size),
                                     method=Image.Resampling.LANCZOS, color="white")
                pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot load image {path}: {error}") from error
        return {"image": pixels / 127.5 - 1, "caption": self.captions[index],
                "label": self.labels[index]}


def load_image_datasets(root, image_size=128, seed=42, val_ratio=0.1):
    """按文件划分 train/val；两者共用 caption -> label 映射。"""
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio must be between 0 and 1")
    paths = find_images(root)
    if len(paths) < 2:
        raise ValueError("Need at least 2 images to split training and validation data")
    caption_bank = list(dict.fromkeys(path.stem.strip() for path in paths))
    random.Random(seed).shuffle(paths)
    val_count = min(len(paths) - 1, max(1, round(len(paths) * val_ratio)))
    train_data = ImageCaptionDataset(paths[val_count:], image_size, caption_bank, "train")
    val_data = ImageCaptionDataset(paths[:val_count], image_size, caption_bank, "val")
    return train_data, val_data


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
    parser.add_argument("--data-dir", help="Image folder; filename without extension is the caption")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output", default="outputs/data-preview.png")
    args = parser.parse_args()
    dataset = (ImageCaptionDataset(find_images(args.data_dir), args.image_size)
               if args.data_dir else ShapesDataset(count=16, image_size=args.image_size))
    items = [dataset[i] for i in range(min(16, len(dataset)))]
    save_grid(torch.stack([item["image"] / 2 + 0.5 for item in items]), args.output)
    Path(args.output).with_suffix(".json").write_text(
        json.dumps([item["caption"] for item in items], indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", args.output)
