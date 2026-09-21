# LoveDA SSDA Semantic Segmentation

本项目提供一套面向 [LoveDA](https://github.com/Junjue-Wang/LoveDA) 遥感语义分割数据集的域适应训练代码。代码以源域有标签数据为基础，支持无标签目标域对抗对齐，并可在第三阶段加入少量目标域有标签样本进行半监督域适应（SSDA）。

## 主要功能

- 七类 LoveDA 语义分割训练、验证和整图预测。
- 支持源域监督学习、无监督域适应（UDA）和带少量目标域标签的 SSDA。
- 分阶段训练：
  - **S1**：源域监督训练。
  - **S2**：加入域对抗对齐，并支持权重 warm-up。
  - **S3**：在指标稳定后启用类别条件对齐，可使用目标域有标签样本作为语义锚点。
- 支持交叉熵、Dice、Focal Tversky、分支监督和门控正则等损失项。
- 按 mIoU、mF1 和 OA 分别保存最佳模型，并可在训练结束后自动生成预测掩膜。
- 当前训练入口按整幅影像读取源域和目标域数据。

## 项目结构

```text
.
├── run_da.py            # 训练入口
├── config/
│   └── default.py       # 数据路径、模型、训练和评估配置
├── common/
│   └── seed.py          # 随机种子与可复现设置
├── data/
│   ├── datasets.py      # 数据集定义
│   └── loader.py        # 数据读取、采样和 DataLoader 构建
├── models/
│   ├── feature.py       # 分割网络、注意力与域门控结构
│   └── domain.py        # 域分类器、随机多线性映射和梯度反转
├── train/
│   ├── loop.py          # S1/S2/S3 训练、验证、保存和预测流程
│   └── utils.py         # 模型参数初始化
├── eval/
│   ├── metrics.py       # 混淆矩阵、mIoU、mF1、OA 等指标
│   └── predict.py       # 整图预测与掩膜保存
└── requirements.txt
```

## 环境要求

- Python 3.8 或更高版本
- 推荐使用支持 CUDA 的 NVIDIA GPU
- PyTorch 的 CUDA 构建需要与本机驱动和 CUDA 环境匹配

建议先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装合适的 PyTorch 版本，再安装其余依赖：

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要指定 CUDA 版本，请先使用 PyTorch 官方提供的安装命令安装 `torch`，然后再执行上面的命令。

## 数据准备

数据集不会随本仓库发布。请自行获取 LoveDA 数据，进行域适应特定数据配置，并遵守数据集的许可和使用条款。

推荐的数据目录结构如下：

```text
datasets/LoveDA/
├── Train/
│   ├── source/
│   │   ├── images/
│   │   └── masks/
│   └── target/
│       └── images/
└── Val/
    ├── source/
    │   ├── images/
    │   └── masks/
    └── target/
        ├── target_val_200/
        │   ├── images/
        │   └── masks/
        ├── target_labeled_100/
        │   ├── images/
        │   └── masks/
        └── target_test_692/
            ├── images/
            └── masks/
```

## 配置

训练前请编辑 `config/default.py`，至少修改以下路径：

```python
CFG["paths"]["checkpoints_dir"]
CFG["paths"]["predict_dir"]

CFG["data"]["source"]["image_dir"]
CFG["data"]["source"]["mask_dir"]
CFG["data"]["target"]["image_dir"]
CFG["data"]["target_val"]["image_dir"]
CFG["data"]["target_val"]["mask_dir"]
CFG["data"]["target_labeled_s3"]["image_dir"]
CFG["data"]["target_labeled_s3"]["mask_dir"]
CFG["data"]["target_test"]["image_dir"]
CFG["data"]["target_test"]["mask_dir"]
```

默认配置使用相对于项目根目录的 `datasets/LoveDA/` 和 `outputs/`。请在项目根目录运行训练命令，或按实际数据位置修改这些相对路径。

若只进行源域监督训练，可设置：

```python
CFG["train"]["enable_uda"] = False
CFG["train"]["enable_s3"] = False
```

若没有目标域有标签样本，可关闭 S3，或将 `s3_align_mode` 改为 `pseudo` 并根据数据调整伪标签阈值。

## 运行训练

在项目根目录执行：

```bash
python run_da.py
```

训练日志会输出当前阶段、各损失项、验证指标、伪标签覆盖率和有效更新次数。

## 输出

默认模型输出目录为 `outputs/checkpoints/`：

```text
outputs/checkpoints/
├── S1/
│   ├── best_by_miou.pkl
│   ├── best_by_mf1.pkl
│   ├── best_by_oa.pkl
│   └── UP1_feature_encoder_final_0.pkl
├── S2/
│   └── ...
├── S3/
│   └── ...
└── UP1_feature_encoder_final_0.pkl
```
