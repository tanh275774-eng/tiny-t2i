"""读取自己训练的 DiT，手写 Euler 采样，再用冻结 VAE 解码。"""

import argparse
import json
from pathlib import Path

import torch

from clip import FrozenCLIP
from dataset import save_grid
from dit import DiTConfig, TinyDiT
from flow import euler_sample
from vae import FrozenVAE

DEFAULT_PROMPTS = [
    "a large red circle in the center on a white background",
    "a large blue circle in the center on a white background",
    "a large red square in the center on a white background",
    "a large red circle on the left on a white background",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", help="Repeat --prompt for multiple captions")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/generated.png")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.steps <= 0 or args.cfg < 0:
        parser.error("steps must be positive and cfg must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use a GPU server or --device cpu")
    output = Path(args.output)
    if output.suffix.lower() != ".png":
        parser.error("Use a .png output")
    if output.exists() or output.with_suffix(".json").exists():
        parser.error("Output exists; choose a new filename")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = DiTConfig(**checkpoint["dit_config"])
    trained = checkpoint["train_args"]
    if args.cfg != 1 and trained["cond_drop"] == 0:
        parser.error("This checkpoint was trained without condition dropout; use --cfg 1")
    model = TinyDiT(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    prompts = args.prompt or DEFAULT_PROMPTS

    encoder = FrozenCLIP(trained["clip_model"], trained["clip_revision"]).to(device)
    text, text_mask = encoder(prompts)
    null_text, null_mask = encoder([""] * len(prompts))
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 所有提示词使用同一份初始噪声，方便只比较文字变化造成的影响。
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(1, config.in_channels, config.latent_size, config.latent_size,
                        generator=generator).repeat(len(prompts), 1, 1, 1).to(device)
    latent = euler_sample(model, noise, text, text_mask, steps=args.steps, cfg=args.cfg,
                          null_text=null_text, null_mask=null_mask)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vae = FrozenVAE(trained["vae_model"], trained["vae_revision"]).to(device)
    # 逐张解码以控制 VAE 峰值显存。
    images = torch.cat([vae.decode(item[None]).cpu() for item in latent])
    save_grid(images, output, columns=min(4, len(prompts)))
    record = {**vars(args), "prompts": prompts, "checkpoint_step": checkpoint["step"],
              "same_initial_noise_for_all_prompts": True}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", output)
    for index, prompt in enumerate(prompts):
        print(f"{index + 1}: {prompt}")


if __name__ == "__main__":
    main()
