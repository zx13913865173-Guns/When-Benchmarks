# 08 — 图表复现（Figures）

对应论文 Results 各节的 Fig. 2 – Fig. 7。全部图由 `make_figures.py` 依据
论文报告的数值重绘，便于核对。

## 1. 图片清单

| 图 | 内容 |
|---|---|
| Fig. 2 | 代理重叠审计：pHash 候选、high / borderline / clean 构成 |
| Fig. 3 | dominant vs secondary recall（60 张非对称图） |
| Fig. 4 | 消融 L0–L4 与 oracle 上界；全局 vs 区域的 precision–recall–F1–coverage 权衡 |
| Fig. 5 | 东西方 gap 随重叠三分位的变化 |
| Fig. 6 | 跨工具对比（CLIP Interrogator / WD14 / DeepDanbooru 的 overall、secondary recall 与东西 gap） |
| Fig. 7 | 描述性汇总：(a) 审计结果，(b) 风格分解结果 |

## 2. 文件

- `make_figures.py`：生成全部图片（英文标注，避免字体问题）。
- `figure_captions.md`：图注与所依据的数值。

## 3. 运行

```bash
python make_figures.py --out_dir ../outputs/figures
```
