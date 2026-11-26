# 复现步骤（End-to-End Reproduction）

从零到产出论文全部数值与图表的逐条命令。命令均以对应模块目录为当前目录执行。

---

## 步骤 0 · 环境准备

```bash
# Python
pip install -r requirements.txt

# R
Rscript -e 'install.packages(c("lme4","lmerTest","emmeans","ggplot2","dplyr","tidyr","purrr","readr","scales"))'
```

确认 GPU 可用（训练与生成阶段）：

```bash
nvidia-smi
```

---

## 步骤 1 · 准备训练数据（375 张）

按 `01_lora_training/prepare_dataset.md` 从 WikiArt / The Met Open Access /
Rijksmuseum 采集 public-domain 图像，每风格 75 张，放入：

```
data/train/{ink_wash,ukiyo_e,cyberpunk,baroque,minimalist}/
```

---

## 步骤 2 · 训练 5 个风格 LoRA

```bash
cd 01_lora_training
bash train_all.sh          # 5 个 LoRA -> ../outputs/lora/<style>/
```

核对：应得到 5 个 `*_sdxl.safetensors`（rank=32、alpha=1.0、3000 steps）。

---

## 步骤 3 · 生成 140 张混合图像

```bash
cd 02_hybrid_generation
python generate_hybrids.py \
    --lora_dir ../outputs/lora \
    --out_dir ../outputs/benchmark_140 \
    --meta ../outputs/generation_metadata.csv
```

核对：`outputs/benchmark_140/` 下应为 140 张 PNG。

---

## 步骤 4 · 构建基准元数据

```bash
cd 03_benchmark
python build_benchmark_metadata.py \
    --config ../02_hybrid_generation/fusion_config.json \
    --seeds ../02_hybrid_generation/seeds.json \
    --out benchmark_metadata.csv
```

预期输出：`total=140 asymmetric=60 equal_weight=60 pure=20`。

---

## 步骤 5 · 专家标注与一致性

```bash
cd 07_annotation
# 4 位专家填写 annotation_template.csv（匿名 ID E1-E4）
python fleiss_kappa.py --annotations annotation_template.csv --out ../outputs/fleiss_kappa.csv
python adjudicate_labels.py --annotations annotation_template.csv --out ground_truth.csv
```

核对：Fleiss κ 应接近 **0.81**。

---

## 步骤 6 · 代理重叠风险审计

```bash
cd 04_proxy_overlap_audit

# 6a. 构建 LAION-2B-en 的 pHash 桶索引（一次性，离线，耗时最长）
python stage1_phash_screen.py build-index \
    --laion_index ../data/laion2b_en_phash.jsonl \
    --index_out ../data/laion2b_en_buckets.pkl

# 6b. 对 375 张训练图做 pHash 候选筛选
python stage1_phash_screen.py screen \
    --train_dir ../data/train \
    --index ../data/laion2b_en_buckets.pkl \
    --out ../outputs/stage1_candidates.csv

# 6c. CLIP 嵌入验证
python stage2_clip_verify.py \
    --candidates ../outputs/stage1_candidates.csv \
    --train_dir ../data/train \
    --out ../outputs/stage2_verified.csv

# 6d. 样式级 / 对级比例与三分位
python pair_overlap_ratio.py \
    --verified ../outputs/stage2_verified.csv \
    --out ../outputs/pair_overlap_ratios.csv
```

核对（论文 §3.1）：候选 16.8%、high 8.5%、borderline 3.5%、合并 12.0%。

---

## 步骤 7 · CLIP Interrogator 评估

```bash
cd 05_interrogator_eval
python run_interrogator.py --benchmark ../outputs/benchmark_140 --out ../outputs/interrogator_prompts.csv
python match_labels.py --prompts ../outputs/interrogator_prompts.csv --out ../outputs/pred_labels.csv
python regional_reinterrogation.py --benchmark ../outputs/benchmark_140 --out ../outputs/prompts_regional.csv
python metrics.py --pred ../outputs/pred_labels.csv \
    --meta ../03_benchmark/benchmark_metadata.csv \
    --truth ../07_annotation/ground_truth.csv \
    --out ../outputs/metrics.csv
```

核对：dominant recall ≈ 0.76、secondary recall ≈ 0.22。

---

## 步骤 8 · 统计分析（R）

```bash
cd 06_statistics
Rscript 00_load_data.R
Rscript 01_permutation_test.R
Rscript 02_mixed_models.R
Rscript 03_fdr_correction.R
```

核对：置换检验枚举 6! = 720 个排列，最小可达 p = 1/360。

---

## 步骤 9 · 生成图表

```bash
cd 08_figures
python make_figures.py --out_dir ../outputs/figures
```

产物：`fig2_audit.png` … `fig7_summary.png`。

---

## 完整数值核对表

| 量 | 论文 | 位置 |
|---|---|---|
| 基准图总数 | 140（120 混合 + 20 纯风格） | Table 1 |
| 非对称图 | 60 | §2.3 |
| 训练图 | 375（75/风格） | §2.2 |
| Fleiss κ | 0.81 | §2.5 |
| pHash 候选 | 63/375 = 16.8% | §3.1 |
| high / borderline | 8.5% / 3.5% | §3.1 |
| 保守合计 | 12.0% | §3.1 |
| dominant recall | 0.76 | §3.4 |
| secondary recall | 0.22 | §3.4 |
| 东西 gap（低/高三分位） | 0.10 / 0.18 | §3.2 |
| 区域重询问 secondary | 0.28 → 0.49 | §3.6 |
