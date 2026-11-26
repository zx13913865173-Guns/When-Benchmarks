# 00 — 环境与依赖（Environment）

> 复现论文所有数值所需的软硬件环境。训练与生成阶段需要 NVIDIA GPU；
> 审计、指标与统计阶段可在 CPU 上完成。

## 1. 硬件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB) ×1（LoRA 训练 / SDXL 生成 / CLIP 推理） |
| CPU | 16 vCPU |
| RAM | 64 GB |
| 存储 | ≥ 500 GB（LAION-2B-en 元数据 + 图像子集 + 生成的 140 图） |

## 2. 软件版本

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| CUDA / cuDNN | 11.8 / 8.9 |
| PyTorch | 2.1.2 |
| diffusers | 0.27.2 |
| kohya-ss/sd-scripts | commit `bd7f4d7`（SDXL 支持分支） |
| CLIP Interrogator | 0.6.0 |
| OpenCLIP | 2.24.0 |
| R | 4.3.2 |
| lme4 | 1.1-35 |

Python 依赖见仓库根目录 `requirements.txt`；R 依赖见下。

## 3. R 包

```r
install.packages(c("lme4", "lmerTest", "emmeans", "ggplot2",
                   "dplyr", "tidyr", "purrr", "readr", "scales"))
```

- `lme4` **必须**为 1.1-35（论文所报数值基于该版本拟合）。
- `lmerTest` 提供 Satterthwaite 自由度近似，用于 `p` 值。

## 4. 模型权重

| 名称 | 来源 |
|---|---|
| SDXL Base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` |
| CLIP ViT-L-14 (openai) | CLIP Interrogator 自动下载 |
| BLIP image-captioning-base | `Salesforce/blip-image-captioning-base` |
| WD14 tagger | `SmilingWolf/wd-v1-4-convnext-tagger-v2`（跨工具验证） |
| DeepDanbooru | `KichangKim/DeepDanbooru`（跨工具验证） |

## 5. 确定性设置

为保证可复现，训练与生成均固定随机种子：

```python
import torch, numpy as np, random
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- LoRA 训练种子：`42`
- 生成基础种子：`42`，每个条件内 5 次重复使用 `42, 43, 44, 45, 46`

## 6. 版本相关的复现注意事项

- CLIP Interrogator 结论基于 **0.6.0**；后续版本配置不同，结论不可直接推广。
- `lme4` 版本影响混合模型的估计量；请锁定 1.1-35。
- pHash 使用 `imagehash.phash`，`hash_size=8`（默认），与论文 Stage 1 一致。
