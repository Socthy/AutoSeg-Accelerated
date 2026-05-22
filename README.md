# AutoSeg-Accelerated

基于 [AutoSeg-SAM2](https://github.com/zrporz/AutoSeg-SAM2) 的加速与工程化改进版本。  
原项目使用 SAM1 做关键帧全图分割、SAM2 做视频跟踪与新物体发现；本仓库在保持该流程的前提下，针对实际数据集与下游训练链路做了优化与扩展。

> **Upstream**: [zrporz/AutoSeg-SAM2](https://github.com/zrporz/AutoSeg-SAM2) — *Automatic full segmentation with SAM1 + SAM2 video tracking.*

---

## 与原版的主要差异（可自行补充）

- `auto-mask-fast.py`：SAM1 权重完整性校验；`masks_update` NMS 迁至 GPU；`--skip_vis` / `--points_per_side` 等加速参数；修复 `final-output` 阶段可视化残留 bug  
- `visulization.py`：支持 `anli2_frame_000184` 等非纯数字帧名排序  
- `autoseg.sh`：路径可配置、`set -e`、默认跳过中间可视化  

---

## 项目概览

|  | 列 1 | 列 2 | 列 3 |
|:--|:--|:--|:--|
| **行 1** | （待填写） | （待填写） | （待填写） |
| **行 2** | （待填写） | （待填写） | （待填写） |

---

## Demo

### 演示 1

<!-- 将视频上传至 GitHub Release / user-attachments 后，把下方链接替换为你的 URL -->

https://github.com/user-attachments/assets/00000000-0000-0000-0000-000000000001

*（待填写：演示 1 说明，例如数据集名称、level、帧数）*

---

### 演示 2

https://github.com/user-attachments/assets/00000000-0000-0000-0000-000000000002

*（待填写：演示 2 说明）*

---

## 环境配置

要求：`python>=3.10`，`torch>=2.3.1`，`torchvision>=0.18.1`，CUDA GPU。

```bash
# 安装 SAM1 / SAM2 子模块
pip install -e submodule/segment-anything-1
pip install -e submodule/segment-anything-2

# 下载权重
cd checkpoints/sam1 && bash download.sh
cd ../sam2 && bash download.sh
```

---

## 数据准备

将视频帧整理为图像序列：

```text
<video_dir>/
  000001.jpg
  000002.jpg
  ...
```

也支持带前缀的帧名（如 `scene_frame_000184.jpg`），需与 AutoSeg 及转换脚本使用同一目录。

---

## 快速开始

### 1. 视频分割（多视角一致跟踪）

```bash
export VIDEO_PATH=/path/to/images
export OUTPUT_DIR=/path/to/output
export LEVELS="large"

bash autoseg.sh
```

或单独运行：

```bash
python auto-mask-fast.py \
  --video_path "$VIDEO_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --level large \
  --detect_stride 10 \
  --points_per_side 16 \
  --skip_vis
```

输出掩码：`$OUTPUT_DIR/large/final-output/mask_XXX.npy`（SAM2 跟踪后的**跨帧一致**实例 mask）。

> 仅 `auto-mask-fast` 中间调试目录（如 `mask_each_frame-sam1`）可能是逐帧独立 SAM1 结果；**下游请使用 `final-output`**。


### 2. 可视化（可选）

```bash
python visulization.py \
  --video_path "$VIDEO_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --level large
```

---

## 目录结构（简要）

```text
AutoSeg-Accelerated/
├── auto-mask-fast.py      # 主流程：SAM1 关键帧 + SAM2 跟踪
├── visulization.py
├── autoseg.sh
├── checkpoints/
├── sam2/
└── submodule/
```

---

## 引用

```bibtex
@software{AutoSeg_SAM2,
  author = {Zrporz},
  title = {AutoSeg-SAM2},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zrporz/AutoSeg-SAM2}
}
```

SAM / SAM2 请同时遵循 [Segment Anything](https://github.com/facebookresearch/segment-anything) 与 [SAM 2](https://github.com/facebookresearch/segment-anything-2) 的相关许可与引用说明。

---

## License

本改进版遵循上游 AutoSeg-SAM2 的 [MIT License](LICENSE)；子模块与权重另受其各自许可约束。
