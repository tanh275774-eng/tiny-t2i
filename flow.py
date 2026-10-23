"""训练和生成共用同一个时间方向：t=0 是噪声，t=1 是数据。"""

import torch


def flow_training_pair(clean, noise, t):
    fraction = t[:, None, None, None]
    mixed = (1 - fraction) * noise + fraction * clean
    target_velocity = clean - noise
    return mixed, target_velocity


@torch.no_grad()
def euler_sample(model, noise, text, text_mask, steps=50, cfg=1.0,
                 null_text=None, null_mask=None):
    if steps <= 0 or cfg < 0:
        raise ValueError("steps must be positive and cfg nonnegative")
    if cfg != 1.0 and (null_text is None or null_mask is None):
        raise ValueError("CFG requires encoded empty prompts")
    latent = noise.clone()
    dt = 1.0 / steps
    for i in range(steps):
        # 最后一次在 1-dt 预测，再更新到 1；没有把时间方向写反。
        t = torch.full((latent.shape[0],), i * dt, device=latent.device)
        velocity = model(latent, t, text, text_mask)
        if cfg != 1.0:
            unconditional = model(latent, t, null_text, null_mask)
            velocity = unconditional + cfg * (velocity - unconditional)
        latent = latent + dt * velocity
    return latent
