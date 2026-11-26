[README.md](https://github.com/user-attachments/files/35439637/README.md)

# When Benchmarks Overlap — 复现工程（Reproducibility Repository）

本仓库为论文 *"When Benchmarks Overlap: Proxy Overlap Risk and Its Limited Role in
Cross-Cultural Style Decomposition Using CLIP Interrogator"* 的完整复现材料，
包含从 LoRA 训练、混合图像生成、140 图基准构建，到代理重叠风险审计、CLIP Interrogator
评估、统计检验与图表生成的**全流程代码与配置**。

仓库结构与论文各节一一对应，可复现论文中报告的全部数值。

---

## 1. 目录结构与论文对应关系

| 目录 | 内容 | 对应论文位置 |
|---|---|---|
| `00_environment/` | 软硬件环境、依赖版本说明 | Methods 通篇 |
| `01_lora_training/` | 5 个风格 LoRA 的训练配置与脚本（kohya-ss/sd-scripts） | §2.2 LoRA training on 375 images |
| `02_hybrid_generation/` | SDXL Base 1.0 多 LoRA 加权融合生成脚本、seed 列表 | §2.3 Hybrid image generation |
| `03_benchmark/` | 140 图基准的组织与元数据（Table 1） | §2.3、Table 1 |
| `04_proxy_overlap_audit/` | 代理重叠风险审计：pHash 候选筛选 + CLIP 嵌入验证 | §2.6、§3.1 |
| `05_interrogator_eval/` | CLIP Interrogator 应用、区域重询问、指标计算 | §2.4、§2.7、§3.4–3.7 |
| `06_statistics/` | R 脚本：置换检验、混合效应模型、FDR 校正 | §2.8、§3.2–3.5 |
| `07_annotation/` | 专家标注模板、Fleiss κ 计算、标签裁定 | §2.5 |
| `08_figures/` | Fig. 2–7 的绘图脚本与图注 | Results 全节 |

---

## 2. 复现顺序（Pipeline）

严格按下列顺序执行，前一步的产物是后一步的输入：

1. **环境准备** → `00_environment/`，安装依赖（`requirements.txt`）。
2. **LoRA 训练** → `01_lora_training/`，为 5 个风格各训练 1 个 LoRA（共 5 个）。
3. **混合图像生成** → `02_hybrid_generation/`，用 6 个融合对 × 权重比 × 5 seeds 生成 120 张混合图 + 20 张纯风格对照 = 140 张。
4. **基准元数据构建** → `03_benchmark/`，整理 140 图与生成元数据。
5. **专家标注与裁定** → `07_annotation/`，得到 ground truth 标签与 Fleiss κ。
6. **代理重叠审计** → `04_proxy_overlap_audit/`，对 375 张 LoRA 训练图做两阶段审计，得到样式级与对级比例、三分位分层。
7. **CLIP Interrogator 评估** → `05_interrogator_eval/`，得到 recall / precision / F1 / coverage、dominant/secondary 指标与消融。
8. **统计分析** → `06_statistics/`，置换检验、混合效应模型、FDR 校正。
9. **图表生成** → `08_figures/`，产出 Fig. 2–7。

---

## 3. 关键超参数

| 参数 | 取值 |
|---|---|
| 基座模型 | SDXL Base 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`) |
| LoRA rank / alpha | 32 / 1.0 |
| 训练步数 / 学习率 / batch | 3000 / 1e-4 / 2 |
| 优化器 | AdamW |
| 训练集规模 | 375 张（75 / 风格），public-domain 或 CC0 |
| 生成分辨率 / CFG / 步数 / 采样器 | 1024×1024 / 7.5 / 30 / DPM++ 2M Karras |
| 权重比 | 0.3:0.7、0.5:0.5、0.7:0.3（0.5:0.5 重复两次） |
| 重复次数 / seeds | 5 次 / 42–46 |
| CLIP Interrogator | v0.6.0，ViT-L-14/openai，classic 模式 |
| pHash 海明距离阈值 | ≤ 10 |
| CLIP 余弦阈值 | ≥0.95 high-similarity；0.90–<0.95 borderline；<0.90 clean |
| 多重比较 | Benjamini–Hochberg FDR |

---

## 4. 数据获取

- **风格训练图**（375 张）：`WikiArt`、`The Met Open Access`、`Rijksmuseum`，
  仅使用 public-domain / CC0 许可图像。见 `01_lora_training/prepare_dataset.md`。
- **代理语料**：`LAION-2B-en`，通过 `img2dataset` 获取并建立 pHash 桶索引，
  见 `04_proxy_overlap_audit/README`（脚本内注释说明规模限制）。
- **生成的 140 张图**：由本仓库 `02_hybrid_generation/` 脚本产生，随仓库附带元数据。

> 伦理说明：生成图像受版权保护，仅用于学术复现；训练图均为公有领域或 CC0；
> 专家标注为匿名、自愿、无补偿，不含个人可识别信息。

---

## 5. 环境安装

```bash
# Python 依赖
pip install -r requirements.txt

# R 依赖
Rscript -e 'install.packages(c("lme4", "lmerTest", "emmeans", "ggplot2", "dplyr", "tidyr"))'
```

详细版本见 `00_environment/environment.md`。

---

## 6. 一键复现

见 `REPRODUCE.md`，其中给出从零到产出全部数值与图表的逐条命令。

- `make_figures.py`：生成全部图片（英文标注，避免字体问题）。
- `figure_captions.md`：图注与所依据的数值。

## 3. 运行

```bash
python make_figures.py --out_dir ../outputs/figures
```
