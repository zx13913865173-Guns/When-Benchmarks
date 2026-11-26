# 训练数据集准备（375 张，75 / 风格）

对应论文 §2.2。训练集共 **375 张**，全部来自公有领域或 CC0 许可来源，
每个风格 75 张。

## 1. 来源与筛选

| 来源 | 获取方式 | 许可要求 |
|---|---|---|
| WikiArt | 公开 API / 页面下载 | public domain |
| The Met Open Access | Met Open Access API (`https://collectionapi.metmuseum.org`) | `isPublicDomain = true` |
| Rijksmuseum | Rijksstudio API | public domain / CC0 |

筛选规则：
1. 仅保留 `isPublicDomain` 或 CC0 标记的图像；
2. 分辨率 ≥ 1024 长边，去重（pHash 相似度 < 0.95）；
3. 每个风格累计到 75 张。

## 2. 目录组织

```
data/train/
├── ink_wash/   *.jpg + *.txt   (75)
├── ukiyo_e/    *.jpg + *.txt   (75)
├── cyberpunk/  *.jpg + *.txt   (75)
├── baroque/    *.jpg + *.txt   (75)
└── minimalist/ *.jpg + *.txt   (75)
```

每个 `.jpg` 对应一个同名 `.txt` 描述文件（kohya 约定）。描述文件为
英文风格描述 + 触发词，例如：

```
ink wash painting style, monochrome brush strokes, xuan paper texture
```

## 3. 与论文原设计的对应

论文说明：相对早期四风格 200 图设计，四个原风格各增加 25 张、第五风格
（minimalist）贡献 75 张，合计 375 张。本仓库即采用 375 张版本。

## 4. 复现校验

训练完成后应得到 5 个 LoRA 文件：

```bash
ls -1 ../outputs/lora/*/*.safetensors
# ../outputs/lora/ink_wash/ink_wash_sdxl.safetensors
# ../outputs/lora/ukiyo_e/ukiyo_e_sdxl.safetensors
# ../outputs/lora/cyberpunk/cyberpunk_sdxl.safetensors
# ../outputs/lora/baroque/baroque_sdxl.safetensors
# ../outputs/lora/minimalist/minimalist_sdxl.safetensors
```
