# 手写一个小型文本生成图片模型

这是用于学习训练流程的项目：冻结预训练 CLIP 和 VAE，从零训练自己定义的小型 DiT。可以使用合成彩色几何图形，也可以读取自己的图片文件夹，把文件名作为提示词；默认分辨率为 128×128。

代码没有调用文生图 Pipeline。DiT 网络、条件注入、Flow Matching 目标、反向传播和 Euler 采样都在本目录显式定义。它不是 Qwen-Image 的复现，也没有通用照片生成能力。

按用户要求仅进行了代码逻辑审阅，未安装完整依赖、未下载模型、未运行单元测试或 GPU 训练；不把这些命令表述为已经实测跑通。

## 文件与阅读顺序

| 文件 | 内容 |
| --- | --- |
| `dataset.py` | 加载以提示词命名的图片，或生成合成形状；支持数据预览 |
| `clip.py` | 冻结 CLIP 文本编码器，提取完整 token 序列与 padding mask |
| `vae.py` | 冻结 VAE，显式处理后验分布、latent 缩放和图像范围 |
| `dit.py` | 自己定义 patch embedding、位置编码、时间嵌入、attention、MLP 和输出层 |
| `flow.py` | 训练插值公式、速度标签、CFG 和 Euler 采样循环 |
| `train.py` | 缓存特征、构造优化器、loss/backward/step、验证与保存 |
| `eval.py` | 加载训练好的 DiT，固定种子改变文本进行对照生成 |

建议先读 `dataset.py`、`vae.py`、`clip.py` 的输入输出，再读 `train.py` 主循环、`flow.py`，最后逐层读 `dit.py`。

## 架构与张量形状

默认训练流程，B 表示 batch size：

```text
图片 [B,3,128,128]，范围 [-1,1]
  -> 冻结 VAE encoder
  -> clean latent [B,4,16,16]，乘 scaling_factor
  -> 和 Gaussian noise 按 t 插值
  -> noisy latent [B,4,16,16]

英文 caption
  -> CLIP tokenizer（最多 77 token，长文本会截断）
  -> 冻结 CLIP text encoder
  -> token features [B,77,512] + mask [B,77]

noisy latent -> 2×2 latent patch -> [B,64,256] + 固定 2D sin-cos 位置编码
t -> sin-cos 时间特征 -> MLP -> 各 block 的 shift、scale、gate
text features -> Linear(512,256) -> cross-attention 的 K、V

6 个 DiT block：图像 self-attention -> 文本 cross-attention -> MLP
  -> 每个 token 预测 2×2×4 个数
  -> unpatchify -> 预测速度 [B,4,16,16]
```

这里的 DiT 是教学简化版，使用 cross-attention 注入文本。它不是 Qwen-Image 的双流 MMDiT，没有 MSRoPE，图像位置采用普通加法 sin-cos 编码。CLIP 自己的文本位置编码由预训练模型处理。

所有 DiT 参数从随机/零初始化开始学习；CLIP、VAE 使用已有权重且不更新。VAE 和 CLIP 输出先缓存，小数据集训练阶段可以把这两个编码器从显存移除。缓存仅存在本次进程内存中，重新运行会重算。

VAE 默认取后验均值 `posterior.mode()`，不是采样；这是为了简化缓存和重建对照。`vae.encode(..., sample_posterior=True)` 保留采样接口，但当前 trainer 使用默认均值。

## 训练与生成对应关系

定义 z 为图像 latent，ε 为噪声：

```python
t = sigmoid(randn(batch))             # logit-normal，t=0 噪声，t=1 数据
x_t = (1 - t) * epsilon + t * z
target_velocity = z - epsilon
prediction = dit(x_t, t, text_features, text_mask)
loss = mse(prediction, target_velocity)
loss.backward()
optimizer.step()
```

生成从纯噪声出发，每步预测速度并前进：

```python
x = epsilon
for i in range(steps):
    t = i / steps
    velocity = dit(x, t, text_features, text_mask)
    x = x + velocity / steps
image = vae.decode(x)
```

实际带 CFG 时，使用空文本预测 v_empty，组合 `v_empty + cfg * (v_text - v_empty)`。训练默认 10% 概率丢弃文本以学习空文本分支；因此空文本仍经过 CLIP 编码，不是全零向量。先用 cfg=1 观察条件学习情况，再试 cfg=2 或 3。

## 环境准备

建议独立的 Python 3.11 Conda 环境，避免改动已有 Qwen 环境：

```bash
conda create -n tiny-t2i python=3.11 -y
conda activate tiny-t2i
# 先安装适合服务器驱动的 CUDA 版 PyTorch >= 2.6。
python -m pip install -r requirements.txt
```

默认加载 `openai/clip-vit-base-patch32` 和 `stabilityai/sd-vae-ft-mse`。训练和推理统一通过 `model_cache.py` 加载：先检查项目根目录的 `model/`，已有完整文件时直接离线使用，不请求远程更新；没有或下载未完成时，默认从 `https://hf-mirror.com` 下载。下载完成后写入文件清单，再次启动会检查文件是否存在及大小是否匹配。仅有空目录不会被当成下载完成。

模型按仓库名和 revision 的哈希分别保存在 `model/<hash>/`；目录中的 `.tiny-t2i-ready.json` 记录来源。`model/` 已加入 `.gitignore`。下载保留 PyTorch 权重、配置和 tokenizer 文件，仓库同时提供 bin/safetensors 时可能下载两种格式。原来用户级 Hugging Face 缓存不会自动搬入这里，因此切换后第一次可能重新下载。

镜像默认生效，也可在启动前显式设置 `export HF_ENDPOINT=https://hf-mirror.com`。外部已有 `HF_ENDPOINT` 会优先保留。仍可用 `--clip-model /path/to/clip`、`--vae-model /path/to/vae` 指定完整本地目录，指定路径只做本地加载。参数 `--clip-revision` 和 `--vae-revision` 可固定仓库 commit；缓存命中后不会检查同名远程分支更新，若要更新模型请指定新的 commit revision。下载失败时保留已下载文件，下次启动会重试。

CLIP 此处只使用文本编码器，不需要 CLIP 图像编码器。为匹配这个英文 CLIP 与合成标签，先用英文提示词。

## 使用自己的图片数据集

把提示词直接用作图片文件名，例如：

```text
data/my-images/
  a red car parked on a street.jpg
  a black cat sitting on a sofa.png
  group-2/
    a white dog running on grass.jpeg
```

支持 JPG、JPEG、PNG、WebP、BMP、TIF、TIFF，扩展名不区分大小写，会递归读取子文件夹。隐藏文件和隐藏文件夹会跳过。`a red car parked on a street.jpg` 对应提示词 `a red car parked on a street`：只去掉最后一个扩展名和首尾空白，不会自动去掉编号、把下划线变为空格，或加入父目录名。同一提示词的多张图片可以放在不同子文件夹下，保持同名。建议直接使用带空格的简短英文文件名；中文文件名可以读取，但当前英文 CLIP 不适合以中文效果为目标，超过 77 token 的文本会截断。

加载时先处理 EXIF 方向，透明部分合成到白底，转换为 RGB，再等比例缩放、居中补白到正方形。默认返回 `image: [3,128,128]`、范围 `[-1,1]`，以及字符串 `caption` 和整数 `label`。`label` 只是该提示词在文本缓存中的索引，不是物体类别。没有随机裁剪或翻转，以免图片与文件名描述不一致。多帧文件只读取首帧；损坏的图片会报告路径并停止。

先预览前 16 张图片及对应提示词：

```bash
python dataset.py --data-dir data/my-images --output outputs/my-data-preview.png
```

开始训练（GPU 0 替换为你实际可用的卡）：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data-dir data/my-images --output runs/my-images-v1
```

至少需要 2 张图片。默认按文件随机留出约 10% 验证，保证训练和验证各至少一张；通过 `--val-ratio 0.1`、`--seed 42` 控制。在文件集合与名称不变时，同一 seed 的划分保持一致。训练/验证共用提示词索引，但不共用文件；不同文件保存了同一张图片的情况不会自动检测，应自行去重。同一提示词可能出现在两个集合，这不代表未见提示词泛化测试。实际划分写入输出目录的 `dataset-split.json`。

指定 `--data-dir` 时读取目录内所有支持的图片，`--samples` / `--val-samples` 仅用于原来的合成数据。`--image-size` 同时控制训练和预览分辨率；默认 VAE 与 patch-size 要求训练分辨率能被 16 整除。每次启动仍需使用新的输出目录，不支持断点续训。

图片 latent 和全部不同提示词的 CLIP 特征缓存在 **CPU 内存**，只把当前 batch 搬到 GPU。默认 CLIP 每条不同提示词约占 154 KiB 特征缓存，1 万条约 1.47 GiB，构建缓存时还会有额外临时内存；这里仍是小中型数据集的教学实现，没有磁盘缓存或流式加载。`data/` 已加入 `.gitignore`，避免误提交数据。

生成时指定自己的提示词：

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint runs/my-images-v1/last.pt --prompt 'a black cat sitting on a sofa' --output outputs/my-cat.png
```

这类新 checkpoint 不传 `--prompt` 时，默认使用保存的至多 4 条训练集提示词；原来形状模型的默认提示词保持不变。更换数据不会自动让小型随机初始化 DiT 获得通用照片生成能力，先用主题较集中、描述准确的数据观察学习情况。

## 以后有空闲 GPU 时的操作顺序

以下命令均在本项目目录中执行；GPU 编号 6 只是示例，必须替换成确认可以使用的卡。代码只使用一张可见 GPU，不实现 DDP。

1. 看配对数据（CPU 即可，不下载模型）。

```bash
python dataset.py --output outputs/data-preview.png
```

得到 4×4 图片网格和按行排列的 caption JSON。形状为圆/方/三角，颜色 6 种，大小 2 种，位置左/中/右，共 108 种 caption。图片还随机抖动坐标和尺寸。没有水平翻转增强，避免把 left 翻成 right 却保留错误文字。

2. 用 8 张图做过拟合学习实验。

```bash
CUDA_VISIBLE_DEVICES=6 python train.py --samples 8 --val-samples 16 --batch-size 8 --steps 1000 --save-every 250 --cond-drop 0 --output runs/overfit
```

目的只是先看少量样本上的 loss 是否下降，1000 步不是保证收敛的承诺。查看 `vae-reconstruction.png` 中相邻的原图/重建图，先了解 VAE 本身的重建误差。过拟合实验只见过少数 caption，生成时应使用这些训练 caption；可用下面的 CPU 命令列出：

```bash
python -c 'from dataset import ShapesDataset; d=ShapesDataset(count=8); print("\n".join(d[i]["caption"] for i in range(8)))'
```

然后把其中一条完整 caption 传给 `eval.py --prompt`。这个 checkpoint 未训练空文本分支，因此只使用 `--cfg 1`。

3. 训练更完整的形状数据。

```bash
CUDA_VISIBLE_DEVICES=6 python train.py --samples 4096 --steps 10000 --output runs/shapes
```

默认 batch=16、hidden=256、depth=6、head=8，全部 FP32。支持 BF16 的 NVIDIA 卡可加 `--precision bf16`；先保持默认值便于阅读与定位问题。如果显存不够，分别降低 `--batch-size`（DiT 训练）和 `--encode-batch-size`（VAE 编码）。不承诺具体显存或耗时，尚未实测。

生成物：

```text
runs/shapes/
  config.json                  训练配置
  vae-reconstruction.png       验证集原图/重建对照
  metrics.jsonl                checkpoint 时的 train/val velocity MSE
  last.pt                      最近一次保存的 DiT 权重、配置、训练步数
```

`last.pt` 在每次保存时更新；尚未实现 optimizer/RNG 保存、断点续训、EMA、学习率调度或分布式训练。重新运行训练必须用新输出目录。这些功能可在核心流程理解后加入。

4. 固定同一初始噪声，对比改变颜色/形状/位置后的生成结果。

```bash
CUDA_VISIBLE_DEVICES=6 python eval.py --checkpoint runs/shapes/last.pt --steps 50 --cfg 1 --output outputs/comparison.png
```

默认四条提示词分别是中心红圆、中心蓝圆、中心红方、左侧红圆。结果按从左到右排列，提示词与种子另存 JSON。

```bash
CUDA_VISIBLE_DEVICES=6 python eval.py --checkpoint runs/shapes/last.pt --prompt 'a small green triangle on the right on a white background' --cfg 2 --output outputs/green-triangle.png
```

`eval.py` 是定性生成对照，不提供 FID 等论文评测指标。训练中固定验证噪声/时间计算的 MSE 也不是感知质量评分。train/val 使用不同随机序列，但都覆盖同一套 caption 组合，不代表组合泛化评测；过拟合少量训练样本也不能证明泛化。

## 逻辑审阅要点

- 图片编码前是 [-1,1]；VAE 解码后转回 [0,1]。
- VAE 编码乘 scaling_factor，解码除同一个值。
- 文本保留 token 序列和 mask，cross-attention 不关注 padding。
- DiT 输出形状与 clean/noise 一样，预测的是速度，不是直接预测 PNG 像素。
- t=0 为噪声，t=1 为数据；target=z-ε，采样步长为正。
- 训练时丢弃条件才能在推理中使用未条件化分支；cfg=1 只走正向条件。
- CLIP/VAE 使用 no_grad 且参数冻结，优化器只有 DiT 参数。
- adaLN gate 和最终输出投影零初始化；最初速度为零，早期不同层梯度会逐步建立，这是设计行为。
- checkpoint 保存实际 DiT 配置与编码器来源，eval 按同一配置恢复。

参考：[CLIPTextModel](https://huggingface.co/docs/transformers/v4.57.1/model_doc/clip)、[AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)、[DiT 原始代码](https://github.com/facebookresearch/DiT)、[PyTorch attention](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)。
