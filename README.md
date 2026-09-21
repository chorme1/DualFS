# LoveDA SSDA Semantic Segmentation

This repository provides domain adaptation training code for remote sensing semantic segmentation on the [LoveDA](https://github.com/Junjue-Wang/LoveDA) dataset. Starting from labeled source-domain data, the code supports adversarial alignment with unlabeled target-domain data and can incorporate a small set of labeled target-domain samples in the third stage for semi-supervised domain adaptation (SSDA).

## Features

- Training, validation, and full-image inference for the seven LoveDA semantic classes.
- Support for source-only supervised learning, unsupervised domain adaptation (UDA), and SSDA with a small labeled target-domain set.
- Three-stage training:
  - **S1**: supervised training on the source domain.
  - **S2**: domain-adversarial alignment with loss-weight warm-up.
  - **S3**: class-conditional alignment activated after the validation metric stabilizes, with labeled target-domain samples available as semantic anchors.
- Support for cross-entropy, Dice, Focal Tversky, branch supervision, and gate regularization losses.
- Separate best checkpoints based on mIoU, mF1, and OA, with optional automatic mask prediction after training.
- The current training entry point loads complete source- and target-domain images rather than sampled patches.

## Project Structure

```text
.
├── run_da.py            # Training entry point
├── config/
│   └── default.py       # Data paths and model, training, and evaluation settings
├── common/
│   └── seed.py          # Random seeds and reproducibility settings
├── data/
│   ├── datasets.py      # Dataset definitions
│   └── loader.py        # Data reading, sampling, and DataLoader construction
├── models/
│   ├── feature.py       # Segmentation network, attention modules, and domain gates
│   └── domain.py        # Domain classifier, randomized multilinear mapping, and gradient reversal
├── train/
│   ├── loop.py          # S1/S2/S3 training, validation, checkpointing, and inference
│   └── utils.py         # Model parameter initialization
├── eval/
│   ├── metrics.py       # Confusion matrix, mIoU, mF1, OA, and related metrics
│   └── predict.py       # Full-image inference and mask export
└── requirements.txt
```

## Requirements

- Python 3.8 or later
- An NVIDIA GPU with CUDA support is recommended
- The PyTorch CUDA build must be compatible with the installed GPU driver and CUDA environment

We recommend installing the appropriate PyTorch build by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) before installing the remaining dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If a specific CUDA version is required, install `torch` with the command provided by the PyTorch website first, and then run the command above.

## Data Preparation

The dataset is not distributed with this repository. Download LoveDA separately, prepare the source and target domains for your domain adaptation setting, and comply with the dataset license and terms of use.

The recommended directory structure is:

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

## Configuration

Before training, edit `config/default.py` and update at least the following paths:

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

The default configuration uses `datasets/LoveDA/` and `outputs/` relative to the project root. Run the training command from the project root, or adjust these paths to match your data location.

For source-only supervised training, use:

```python
CFG["train"]["enable_uda"] = False
CFG["train"]["enable_s3"] = False
```

If no labeled target-domain samples are available, disable S3 or set `s3_align_mode` to `pseudo` and adjust the pseudo-label confidence thresholds for your data.

## Training

Run the following command from the project root:

```bash
python run_da.py
```

The training log reports the current stage, individual loss terms, validation metrics, pseudo-label coverage, and the number of valid optimization updates.

## Outputs

By default, model checkpoints are written to `outputs/checkpoints/`:

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
