import os
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.seed import same_seeds
from data.datasets import SegmentationImageDataset
from data.loader import _collect_labeled_pairs, _read_image, _read_mask, get_eval_loader, get_train_test_loader_multi, get_unlabeled_loader, select_labeled_pairs
from eval.metrics import evaluate_fusion_classifier, format_per_class, metrics_from_confusion
from eval.predict import predict_multispectral_images_batch
from models.domain import DomainClassifier, RandomLayer
from models.feature import FeatureNetwork
from train.utils import weights_init


def _ensure_dirs(paths):
    for key in ("checkpoints_dir",):
        if key in paths:
            os.makedirs(paths[key], exist_ok=True)


def _valid_one_hot(labels, num_classes, ignore_index):
    valid = labels != ignore_index
    safe_labels = labels.clone()
    safe_labels[~valid] = 0
    safe_labels = safe_labels.clamp(min=0, max=num_classes - 1)
    one_hot = torch.nn.functional.one_hot(safe_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return one_hot * valid.unsqueeze(1).float(), valid.unsqueeze(1).float()


def _dice_loss(logits, labels, eps=1e-6, ignore_index=255, class_weights=None):
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    target, valid = _valid_one_hot(labels, num_classes, ignore_index)
    probs = probs * valid

    dims = (0, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice_loss = 1.0 - (2.0 * intersection + eps) / (union + eps)

    present = target.sum(dim=dims) > 0
    if class_weights is not None:
        weights = class_weights.to(logits.device).float()
        weights = weights / weights.mean().clamp_min(eps)
        dice_loss = dice_loss * weights
        return dice_loss[present].sum() / weights[present].sum().clamp_min(eps)
    return dice_loss[present].mean() if present.any() else dice_loss.mean()


def _focal_tversky_loss(logits, labels, alpha=0.3, beta=0.7, gamma=1.5, eps=1e-6, ignore_index=255, class_weights=None):
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    target, valid = _valid_one_hot(labels, num_classes, ignore_index)
    probs = probs * valid

    dims = (0, 2, 3)
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target) * valid).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    focal_tversky = torch.pow(1.0 - tversky, gamma)

    present = target.sum(dim=dims) > 0
    if class_weights is not None:
        weights = class_weights.to(logits.device).float()
        weights = weights / weights.mean().clamp_min(eps)
        focal_tversky = focal_tversky * weights
        return focal_tversky[present].sum() / weights[present].sum().clamp_min(eps)
    return focal_tversky[present].mean() if present.any() else focal_tversky.mean()


def _effective_da_lambda(cfg, epoch_index):
    base_lambda = cfg["train"]["da_lambda"]
    stage1_epochs = cfg["train"].get("stage1_epochs", 0)
    stage2_warmup_epochs = cfg["train"].get("stage2_warmup_epochs", 0)

    current_epoch = epoch_index + 1
    if current_epoch <= stage1_epochs:
        return 0.0, "S1"

    if stage2_warmup_epochs > 0:
        progress = (current_epoch - stage1_epochs) / float(stage2_warmup_epochs)
        warmup_scale = min(1.0, max(0.0, progress))
    else:
        warmup_scale = 1.0
    return base_lambda * warmup_scale, "S2"


def _effective_s3_lambda(cfg, epoch_index):
    s3_base = cfg["train"].get("s3_cc_lambda", 0.0)
    s3_start = cfg["train"].get("s3_start_epoch", 0)
    s3_warmup = cfg["train"].get("s3_warmup_epochs", 0)
    current_epoch = epoch_index + 1
    if s3_base <= 0.0 or current_epoch <= s3_start:
        return 0.0
    if s3_warmup <= 0:
        return s3_base
    progress = (current_epoch - s3_start) / float(s3_warmup)
    return s3_base * min(1.0, max(0.0, progress))


def _pseudo_conf_threshold(cfg, epoch_index):
    start_epoch = cfg["train"].get("s3_start_epoch", 0)
    start_thr = cfg["train"].get("pseudo_conf_start", 0.95)
    end_thr = cfg["train"].get("pseudo_conf_end", 0.80)
    decay_epochs = cfg["train"].get("pseudo_conf_decay_epochs", 0)
    current_epoch = epoch_index + 1
    if current_epoch <= start_epoch:
        return start_thr
    if decay_epochs <= 0:
        return end_thr
    ratio = (current_epoch - start_epoch) / float(decay_epochs)
    ratio = min(1.0, max(0.0, ratio))
    return start_thr + ratio * (end_thr - start_thr)


def _is_metric_stable(history, window=5, min_mean=0.05, max_std=0.02):
    if len(history) < max(1, window):
        return False
    recent = np.array(history[-window:], dtype=np.float32)
    return float(recent.mean()) >= float(min_mean) and float(recent.std()) <= float(max_std)


def _class_conditional_align_loss(
    source_feat,
    source_mask,
    target_feat,
    target_logits,
    class_num,
    conf_threshold=0.9,
    min_pixels=64,
    ignore_index=255,
):
    eps = 1e-12
    source_vec = source_feat.permute(0, 2, 3, 1).reshape(-1, source_feat.size(1))
    source_lbl = source_mask.reshape(-1)

    target_prob = torch.softmax(target_logits, dim=1)
    target_conf, target_pseudo = torch.max(target_prob, dim=1)
    target_valid = target_conf > conf_threshold
    target_vec = target_feat.permute(0, 2, 3, 1).reshape(-1, target_feat.size(1))
    target_lbl = target_pseudo.reshape(-1)
    target_valid = target_valid.reshape(-1)

    per_class_losses = []
    matched_classes = 0
    for cls_idx in range(class_num):
        src_idx = (source_lbl == cls_idx) & (source_lbl != ignore_index)
        tar_idx = (target_lbl == cls_idx) & target_valid
        if src_idx.sum().item() < min_pixels or tar_idx.sum().item() < min_pixels:
            continue
        src_center = source_vec[src_idx].mean(dim=0)
        tar_center = target_vec[tar_idx].mean(dim=0)
        per_class_losses.append(((src_center - tar_center) ** 2).mean())
        matched_classes += 1

    total_target_pixels = float(target_valid.numel())
    valid_target_pixels = float(target_valid.sum().item())
    pseudo_coverage = valid_target_pixels / (total_target_pixels + eps)
    pseudo_ratio = valid_target_pixels / (total_target_pixels + eps)

    if matched_classes == 0:
        zero = source_feat.sum() * 0.0
        return zero, 0, pseudo_coverage, pseudo_ratio, False

    cc_loss = torch.stack(per_class_losses).mean()
    return cc_loss, matched_classes, pseudo_coverage, pseudo_ratio, True


def _class_conditional_labeled_align_loss(
    source_feat,
    source_mask,
    target_feat,
    target_mask,
    class_num,
    min_pixels=64,
    ignore_index=255,
):
    eps = 1e-12
    source_vec = source_feat.permute(0, 2, 3, 1).reshape(-1, source_feat.size(1))
    source_lbl = source_mask.reshape(-1)
    target_vec = target_feat.permute(0, 2, 3, 1).reshape(-1, target_feat.size(1))
    target_lbl = target_mask.reshape(-1)
    target_valid = target_lbl != ignore_index

    per_class_losses = []
    matched_classes = 0
    for cls_idx in range(class_num):
        src_idx = (source_lbl == cls_idx) & (source_lbl != ignore_index)
        tar_idx = (target_lbl == cls_idx) & target_valid
        if src_idx.sum().item() < min_pixels or tar_idx.sum().item() < min_pixels:
            continue
        src_center = source_vec[src_idx].mean(dim=0)
        tar_center = target_vec[tar_idx].mean(dim=0)
        per_class_losses.append(((src_center - tar_center) ** 2).mean())
        matched_classes += 1

    total_target_pixels = float(target_valid.numel())
    valid_target_pixels = float(target_valid.sum().item())
    labeled_coverage = valid_target_pixels / (total_target_pixels + eps)
    labeled_ratio = valid_target_pixels / (total_target_pixels + eps)

    if matched_classes == 0:
        zero = source_feat.sum() * 0.0
        return zero, 0, labeled_coverage, labeled_ratio, False

    cc_loss = torch.stack(per_class_losses).mean()
    return cc_loss, matched_classes, labeled_coverage, labeled_ratio, True


def _next_or_restart(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def _path_numeric_id(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    try:
        return int(stem)
    except ValueError:
        return None


def _build_fixed_id_labeled_loader(
    image_dir,
    mask_dir,
    class_num,
    expected_bands,
    start_id,
    end_id,
    batch_size,
    normalize=False,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=None,
    )
    fixed_image_paths = []
    fixed_mask_paths = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        numeric_id = _path_numeric_id(image_path)
        if numeric_id is not None and int(start_id) <= numeric_id <= int(end_id):
            fixed_image_paths.append(image_path)
            fixed_mask_paths.append(mask_path)

    if not fixed_image_paths:
        raise RuntimeError(
            "No target labeled SSDA samples found in {} for id range {:05d}-{:05d}".format(
                image_dir,
                int(start_id),
                int(end_id),
            )
        )

    dataset = SegmentationImageDataset(
        fixed_image_paths,
        fixed_mask_paths,
        image_reader=lambda path: _read_image(path, normalize, expected_bands),
        mask_reader=lambda path: _read_mask(path, class_num),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0), dataset


def _build_random_labeled_loader(
    image_dir,
    mask_dir,
    class_num,
    expected_bands,
    num_labeled,
    seed,
    batch_size,
    normalize=False,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=None,
    )
    image_paths, mask_paths = select_labeled_pairs(
        image_paths,
        mask_paths,
        num_labeled=num_labeled,
        seed=seed,
    )
    dataset = SegmentationImageDataset(
        image_paths,
        mask_paths,
        image_reader=lambda path: _read_image(path, normalize, expected_bands),
        mask_reader=lambda path: _read_mask(path, class_num),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0), dataset


def _build_labeled_val_test_loaders(
    image_dir,
    mask_dir,
    class_num,
    expected_bands,
    val_count,
    batch_size,
    normalize=False,
    target_labeled_count=0,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=None,
    )
    pairs = list(zip(image_paths, mask_paths))
    target_labeled_count = int(target_labeled_count or 0)
    val_count = int(val_count)
    if target_labeled_count < 0:
        raise RuntimeError("target_val.target_labeled_count must be >= 0, got {}".format(target_labeled_count))
    if val_count <= 0:
        raise RuntimeError("target_val.val_count must be positive, got {}".format(val_count))
    if target_labeled_count + val_count >= len(pairs):
        raise RuntimeError(
            "target_labeled_count + val_count must be smaller than total target val samples: "
            "{} + {} >= {}".format(target_labeled_count, val_count, len(pairs))
        )
    target_labeled_pairs = pairs[:target_labeled_count]
    val_pairs = pairs[target_labeled_count : target_labeled_count + val_count]
    test_pairs = pairs[target_labeled_count + val_count :]

    def _make_loader(split_pairs, shuffle=False):
        dataset = SegmentationImageDataset(
            [pair[0] for pair in split_pairs],
            [pair[1] for pair in split_pairs],
            image_reader=lambda path: _read_image(path, normalize, expected_bands),
            mask_reader=lambda path: _read_mask(path, class_num),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0), dataset

    target_labeled_loader = None
    target_labeled_dataset = None
    if target_labeled_pairs:
        target_labeled_loader, target_labeled_dataset = _make_loader(target_labeled_pairs, shuffle=True)
    val_loader, val_dataset = _make_loader(val_pairs)
    test_loader, test_dataset = _make_loader(test_pairs)
    return (
        target_labeled_loader,
        target_labeled_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )


def _build_labeled_loader_excluding_id_range(
    image_dir,
    mask_dir,
    class_num,
    expected_bands,
    exclude_start_id,
    exclude_end_id,
    batch_size,
    normalize=False,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=None,
    )
    kept_image_paths = []
    kept_mask_paths = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        numeric_id = _path_numeric_id(image_path)
        if numeric_id is not None and int(exclude_start_id) <= numeric_id <= int(exclude_end_id):
            continue
        kept_image_paths.append(image_path)
        kept_mask_paths.append(mask_path)

    if not kept_image_paths:
        raise RuntimeError(
            "No labeled validation samples remain after excluding id range {:05d}-{:05d} from {}".format(
                int(exclude_start_id),
                int(exclude_end_id),
                image_dir,
            )
        )

    dataset = SegmentationImageDataset(
        kept_image_paths,
        kept_mask_paths,
        image_reader=lambda path: _read_image(path, normalize, expected_bands),
        mask_reader=lambda path: _read_mask(path, class_num),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), dataset


def _build_labeled_val_test_loaders_excluding_id_range(
    image_dir,
    mask_dir,
    class_num,
    expected_bands,
    exclude_start_id,
    exclude_end_id,
    val_count,
    batch_size,
    normalize=False,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=None,
    )
    kept_pairs = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        numeric_id = _path_numeric_id(image_path)
        if numeric_id is not None and int(exclude_start_id) <= numeric_id <= int(exclude_end_id):
            continue
        kept_pairs.append((image_path, mask_path))

    val_count = int(val_count)
    if val_count <= 0:
        raise RuntimeError("target_val_count must be positive when target test evaluation is enabled.")
    if len(kept_pairs) <= val_count:
        raise RuntimeError(
            "Not enough target heldout samples to split val/test: kept={}, val_count={}".format(
                len(kept_pairs),
                val_count,
            )
        )

    val_pairs = kept_pairs[:val_count]
    test_pairs = kept_pairs[val_count:]

    def _make_loader(pairs):
        dataset = SegmentationImageDataset(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            image_reader=lambda path: _read_image(path, normalize, expected_bands),
            mask_reader=lambda path: _read_mask(path, class_num),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), dataset

    val_loader, val_dataset = _make_loader(val_pairs)
    test_loader, test_dataset = _make_loader(test_pairs)
    return val_loader, val_dataset, test_loader, test_dataset


def _run_source_mask_diagnostics(dataset, class_num, foreground_index=1):
    class_pixel_counts = np.zeros(class_num, dtype=np.int64)
    unique_values = set()

    for mask_path in dataset.mask_paths:
        mask = dataset.mask_reader(mask_path)
        vals, cnts = np.unique(mask, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            unique_values.add(int(v))
            if 0 <= int(v) < class_num:
                class_pixel_counts[int(v)] += int(c)

    print("=== Source Mask Diagnostics ===")
    print("mask unique values:", sorted(unique_values))
    print("class pixel counts:", class_pixel_counts.tolist())


def _save_alignment_samples(dataset, out_dir, num_samples=10, seed=0, foreground_index=1):
    os.makedirs(out_dir, exist_ok=True)
    total = len(dataset)
    if total == 0:
        return
    rng = np.random.RandomState(seed)
    sample_count = min(num_samples, total)
    indices = rng.choice(total, size=sample_count, replace=False)

    for i, idx in enumerate(indices.tolist(), start=1):
        image_path = dataset.image_paths[idx]
        mask_path = dataset.mask_paths[idx]
        image = dataset.image_reader(image_path)
        mask = dataset.mask_reader(mask_path)

        if image.shape[2] >= 3:
            rgb = image[:, :, :3]
        else:
            rgb = np.repeat(image[:, :, :1], 3, axis=2)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255.0).astype(np.uint8)

        mask_vis = np.zeros_like(rgb_u8)
        fg = mask == foreground_index
        mask_vis[:, :, 0][fg] = 255
        mask_vis[:, :, 1][fg] = 255
        mask_vis[:, :, 2][fg] = 255

        overlay = rgb_u8.copy()
        overlay[:, :, 0][fg] = 255
        overlay[:, :, 1][fg] = (overlay[:, :, 1][fg] * 0.4).astype(np.uint8)
        overlay[:, :, 2][fg] = (overlay[:, :, 2][fg] * 0.4).astype(np.uint8)

        panel = np.concatenate([rgb_u8, mask_vis, overlay], axis=1)
        save_name = "sample_{:02d}.png".format(i)
        imageio.imwrite(os.path.join(out_dir, save_name), panel)

    print("saved alignment samples to:", out_dir)


def _print_batch_fg_ratio(dataset, batch_size, max_batches=10, foreground_index=1, ignore_index=2):
    diag_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("=== Source Batch Foreground Ratio (valid pixels, first {} batches) ===".format(max_batches))
    for batch_idx, (_, masks) in enumerate(diag_loader):
        if batch_idx >= max_batches:
            break
        valid = masks != ignore_index
        fg = (masks == foreground_index) & valid
        valid_pixels = float(valid.sum().item())
        fg_ratio = float(fg.sum().item()) / max(valid_pixels, 1e-12)
        print("batch {:>3d} fg_ratio {:.6f}".format(batch_idx + 1, fg_ratio))


def _evaluate_labeled_split(
    split_name,
    feature_encoder,
    image_dir,
    mask_dir,
    class_num,
    data_cfg,
    cfg,
    device,
    domain="source",
):
    feature_encoder.eval()
    eval_loader = get_eval_loader(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=data_cfg["limit_images"],
        normalize=data_cfg["normalize"],
        expected_bands=data_cfg["input_bands"],
        batch_size=cfg["eval"].get("val_batch_size", 1),
    )
    m_f1, m_iou, conf = evaluate_fusion_classifier(
        feature_encoder,
        eval_loader,
        class_num,
        device,
        domain=domain,
    )
    metrics = metrics_from_confusion(conf)
    print(
        "[{}] domain={} | images: {} | image_dir: {} | mask_dir: {}".format(
            split_name,
            domain,
            len(eval_loader.dataset),
            image_dir,
            mask_dir,
        )
    )
    print(
        "[{}] mF1: {:.4f} | mIoU: {:.4f} | OA: {:.4f}".format(
            split_name,
            m_f1,
            m_iou,
            metrics["oa"],
        )
    )
    print("[{}] per-class IoU | {}".format(split_name, format_per_class(metrics["iou"])))
    print("[{}] per-class F1  | {}".format(split_name, format_per_class(metrics["f1"])))
    return m_f1, m_iou, conf


def _evaluate_named_loader(split_name, feature_encoder, dataloader, class_num, device, domain="source"):
    feature_encoder.eval()
    m_f1, m_iou, conf = evaluate_fusion_classifier(
        feature_encoder,
        dataloader,
        class_num,
        device,
        domain=domain,
    )
    metrics = metrics_from_confusion(conf)
    print(
        "[{}] domain={} | images: {}".format(
            split_name,
            domain,
            len(dataloader.dataset),
        )
    )
    print(
        "[{}] mF1: {:.4f} | mIoU: {:.4f} | OA: {:.4f}".format(
            split_name,
            m_f1,
            m_iou,
            metrics["oa"],
        )
    )
    print("[{}] per-class IoU | {}".format(split_name, format_per_class(metrics["iou"])))
    print("[{}] per-class F1  | {}".format(split_name, format_per_class(metrics["f1"])))
    return m_f1, m_iou, conf


def run(cfg):
    device_cfg = cfg["device"]
    if device_cfg.get("use_cuda", True) and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_cfg.get('cuda_index', 0)}")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    same_seeds(cfg["seed"])
    _ensure_dirs(cfg["paths"])

    class_num = cfg["data"]["class_num"]
    data_cfg = cfg["data"]
    enable_uda = cfg["train"].get("enable_uda", True)

    source_train_loader, source_eval_loader, _ = get_train_test_loader_multi(
        image_dir=data_cfg["source"]["image_dir"],
        mask_dir=data_cfg["source"]["mask_dir"],
        class_num=class_num,
        limit_images=data_cfg["limit_images"],
        normalize=data_cfg["normalize"],
        expected_bands=data_cfg["input_bands"],
        train_all_samples=cfg["train"].get("use_all_source_samples", True),
        train_batch_size=cfg["train"].get("source_batch_size", 1),
        test_batch_size=cfg["eval"].get("val_batch_size", 1),
        foreground_aware_sampling=cfg["train"].get("foreground_aware_sampling", False),
        foreground_class_index=cfg["train"].get("foreground_class_index", 1),
        fg_min_pixels_per_image=cfg["train"].get("fg_min_pixels_per_image", 1),
        fg_per_batch=cfg["train"].get("fg_per_batch", 1),
        ignore_index=cfg["train"].get("ignore_index", 2),
    )

    target_metric_loader = None
    target_metric_name = None
    target_test_loader = None
    target_labeled_loader = None
    target_labeled_dataset = None
    target_labeled_iter = None
    if cfg["eval"].get("use_val", False) and "target_val" in data_cfg:
        target_val_cfg = data_cfg["target_val"]
        target_metric_loader = get_eval_loader(
            image_dir=target_val_cfg["image_dir"],
            mask_dir=target_val_cfg["mask_dir"],
            class_num=class_num,
            limit_images=data_cfg["limit_images"],
            normalize=data_cfg["normalize"],
            expected_bands=data_cfg["input_bands"],
            batch_size=cfg["eval"].get("val_batch_size", 1),
        )
        target_metric_name = "target_val"
        print(
            "[Target Val] enabled | images: {} | image_dir: {} | mask_dir: {}".format(
                len(target_metric_loader.dataset),
                target_val_cfg["image_dir"],
                target_val_cfg["mask_dir"],
            )
        )
    if cfg["eval"].get("use_val", False) and "target_test" in data_cfg:
        target_test_cfg = data_cfg["target_test"]
        target_test_loader = get_eval_loader(
            image_dir=target_test_cfg["image_dir"],
            mask_dir=target_test_cfg["mask_dir"],
            class_num=class_num,
            limit_images=data_cfg["limit_images"],
            normalize=data_cfg["normalize"],
            expected_bands=data_cfg["input_bands"],
            batch_size=cfg["eval"].get("val_batch_size", 1),
        )
        print(
            "[Target Test] enabled | images: {} | image_dir: {} | mask_dir: {}".format(
                len(target_test_loader.dataset),
                target_test_cfg["image_dir"],
                target_test_cfg["mask_dir"],
            )
        )

    debug_cfg = cfg.get("debug", {})
    if debug_cfg.get("enable_data_diagnostics", True):
        fg_idx = 1 if class_num > 1 else 0
        if hasattr(source_train_loader.dataset, "mask_paths"):
            _run_source_mask_diagnostics(source_train_loader.dataset, class_num=class_num, foreground_index=fg_idx)
            if debug_cfg.get("save_alignment_samples", True):
                align_dir = os.path.join(cfg["paths"]["checkpoints_dir"], "debug_alignment")
                _save_alignment_samples(
                    dataset=source_train_loader.dataset,
                    out_dir=align_dir,
                    num_samples=debug_cfg.get("alignment_samples", 10),
                    seed=cfg.get("seed", 0),
                    foreground_index=fg_idx,
                )
        else:
            print("=== Source Mask Diagnostics ===")
            print("Skipped full-image diagnostics in patch-training mode.")
        if int(debug_cfg.get("batch_fg_ratio_batches", 0)) > 0:
            _print_batch_fg_ratio(
                dataset=source_train_loader.dataset,
                batch_size=cfg["train"].get("source_batch_size", 1),
                max_batches=debug_cfg.get("batch_fg_ratio_batches", 10),
                foreground_index=fg_idx,
            )

    target_loader = None
    if enable_uda:
        target_loader = get_unlabeled_loader(
            image_dir=data_cfg["target"]["image_dir"],
            limit_images=data_cfg["limit_images"],
            normalize=data_cfg["normalize"],
            batch_size=cfg["train"]["target_batch_size"],
            expected_bands=data_cfg["input_bands"],
        )
        ssda_cfg = data_cfg.get("target_labeled_s3", None)
        if (
            target_labeled_loader is None
            and ssda_cfg
            and cfg["train"].get("s3_align_mode", "pseudo") == "target_labeled"
        ):
            target_labeled_loader, target_labeled_dataset = _build_random_labeled_loader(
                image_dir=ssda_cfg["image_dir"],
                mask_dir=ssda_cfg["mask_dir"],
                class_num=class_num,
                expected_bands=data_cfg["input_bands"],
                num_labeled=ssda_cfg.get("num_labeled", None),
                seed=ssda_cfg.get("seed", cfg.get("seed", 0)),
                batch_size=cfg["train"].get("target_labeled_batch_size", cfg["train"].get("target_batch_size", 1)),
                normalize=data_cfg["normalize"],
            )
            target_labeled_iter = iter(target_labeled_loader)
            print(
                "[SSDA-S3] target labeled semantic anchors enabled | images: {} | seed: {}".format(
                    len(target_labeled_dataset),
                    int(ssda_cfg.get("seed", cfg.get("seed", 0))),
                )
            )
        elif target_labeled_loader is not None and cfg["train"].get("s3_align_mode", "pseudo") == "target_labeled":
            print(
                "[SSDA-S3] target labeled semantic anchors enabled from target_val split | images: {}".format(
                    len(target_labeled_dataset),
                )
            )

    feature_encoder = FeatureNetwork(
        feature_dim=cfg["model"]["feature_dim"],
        src_input_dim=cfg["model"]["src_input_dim"],
        tar_input_dim=cfg["model"]["tar_input_dim"],
        n_dim=cfg["model"]["n_dim"],
        class_num=class_num,
        e1_channels=cfg["model"]["e1_channels"],
    )
    domain_classifier = DomainClassifier()
    random_layer = RandomLayer([cfg["model"]["feature_dim"], class_num], cfg["model"]["random_layer_dim"])

    feature_encoder.apply(weights_init)
    domain_classifier.apply(weights_init)

    feature_encoder.to(device)
    domain_classifier.to(device)
    random_layer.to(device)

    feature_optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=cfg["train"]["learning_rate"])
    domain_optimizer = torch.optim.Adam(domain_classifier.parameters(), lr=cfg["train"]["learning_rate"])

    class_weights = cfg["train"].get("class_weights", None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    ignore_index = cfg["train"].get("ignore_index", 2)
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)
    ft_alpha = cfg["train"].get("focal_tversky_alpha", 0.7)
    ft_beta = cfg["train"].get("focal_tversky_beta", 0.3)
    ft_gamma = cfg["train"].get("focal_tversky_gamma", 1.33)
    ft_weight = cfg["train"].get("focal_tversky_weight", 1.0)
    dice_weight = cfg["train"].get("dice_loss_weight", 0.0)

    if enable_uda:
        print("Training (Full-image UNet + Dual-Branch DA, full-source-per-epoch)...")
    else:
        print("Training (Pure Source Supervised, no UDA)...")
    source_batches = len(source_train_loader)
    if enable_uda:
        target_batches = len(target_loader)
        if source_batches == 0 or target_batches == 0:
            raise RuntimeError("Empty dataloader: please check source/target image directories.")
        print(f"Source batches per epoch: {source_batches} | Target batches: {target_batches}")
        print(
            "Source images: {} | Target images: {}".format(
                len(source_train_loader.dataset), len(target_loader.dataset)
            )
        )
    else:
        if source_batches == 0:
            raise RuntimeError("Empty source dataloader: please check source image directories.")
        print(f"Source batches per epoch: {source_batches}")
        print("Source patches: {} | Fixed eval patches: {}".format(
            len(source_train_loader.dataset), len(source_eval_loader.dataset)
        ))

    source_iter = iter(source_train_loader)
    target_iter = iter(target_loader) if enable_uda else None
    train_start = time.time()
    nan_skip_count = 0

    max_epoch = int(cfg["train"].get("max_epoch", 0))
    if max_epoch <= 0:
        raise ValueError("train.max_epoch must be > 0")

    global_step = 0
    miou_history = []
    s3_active_epochs = 0

    stage_names = ("S1", "S2", "S3")
    stage_best = {
        stage: {
            "miou": {"value": -1.0, "epoch": 0, "filename": "best_by_miou.pkl"},
            "mf1": {"value": -1.0, "epoch": 0, "filename": "best_by_mf1.pkl"},
            "oa": {"value": -1.0, "epoch": 0, "filename": "best_by_oa.pkl"},
        }
        for stage in stage_names
    }
    stage_seen = {stage: 0 for stage in stage_names}
    for stage in stage_names:
        os.makedirs(os.path.join(cfg["paths"]["checkpoints_dir"], stage), exist_ok=True)

    for epoch in range(max_epoch):
        epoch_sums = {
            "cls": 0.0,
            "dice": 0.0,
            "ft": 0.0,
            "sep": 0.0,
            "domain": 0.0,
            "tgt_sup": 0.0,
            "s3_tgt_sup": 0.0,
            "cc": 0.0,
            "sparse": 0.0,
            "excl": 0.0,
            "total": 0.0,
            "cc_cls": 0.0,
            "pseudo_cov": 0.0,
            "pseudo_fg_ratio": 0.0,
            "cc_gate_on": 0.0,
            "cc_w_eff": 0.0,
        }
        epoch_updates = 0

        effective_da_lambda, stage_tag = _effective_da_lambda(cfg, epoch)
        s3_lambda_candidate = _effective_s3_lambda(cfg, epoch)
        if not enable_uda:
            effective_da_lambda = 0.0
            s3_lambda_candidate = 0.0
            stage_tag = "SRC"

        pseudo_thr = _pseudo_conf_threshold(cfg, epoch)

        s3_lambda = 0.0
        enable_s3 = cfg["train"].get("enable_s3", True)
        s3_max_epochs = int(cfg["train"].get("s3_max_epochs", 6))
        s3_stable = _is_metric_stable(
            miou_history,
            window=int(cfg["train"].get("s3_stability_window", 5)),
            min_mean=float(cfg["train"].get("s3_stability_min_miou", 0.05)),
            max_std=float(cfg["train"].get("s3_stability_max_std", 0.02)),
        )

        if (
            enable_uda
            and enable_s3
            and s3_lambda_candidate > 0.0
            and s3_stable
            and s3_active_epochs < s3_max_epochs
        ):
            s3_lambda = s3_lambda_candidate
            stage_tag = "S3"

        for _ in range(source_batches):
            global_step += 1

            (source_data, source_mask), source_iter = _next_or_restart(source_iter, source_train_loader)
            if enable_uda:
                target_data, target_iter = _next_or_restart(target_iter, target_loader)
            else:
                target_data = None

            source_data = source_data.to(device)
            source_mask = source_mask.to(device)
            if enable_uda:
                target_data = target_data.to(device)

            source_data = torch.nan_to_num(source_data, nan=0.0, posinf=0.0, neginf=0.0)
            if enable_uda:
                target_data = torch.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
            source_mask = torch.nan_to_num(source_mask.float(), nan=0.0, posinf=0.0, neginf=0.0).long()

            source_out = feature_encoder(source_data, domain="source")

            source_logits = source_out["fusion_logits"]
            source_features = source_out["e2_feat"]
            target_source_sup_weight = float(cfg["train"].get("target_source_sup_weight", 0.0))
            if enable_uda and target_source_sup_weight > 0.0:
                target_on_source_out = feature_encoder(source_data, domain="target")
            else:
                target_on_source_out = None

            if enable_uda:
                target_out = feature_encoder(target_data, domain="target")
                target_logits = target_out["fusion_logits"]
                target_features = target_out["e2_feat"]
                source_softmax = torch.softmax(source_logits, dim=1).mean(dim=(2, 3))
                target_softmax = torch.softmax(target_logits, dim=1).mean(dim=(2, 3))
                domain_features = torch.cat([source_features, target_features], dim=0)
                domain_outputs = torch.cat([source_softmax, target_softmax], dim=0)
                domain_label = torch.zeros(domain_features.size(0), 1, device=device)
                domain_label[: source_features.size(0)] = 1
                random_out = random_layer([domain_features, domain_outputs])
                domain_logits = domain_classifier(random_out, global_step)
                domain_loss = domain_criterion(domain_logits, domain_label)
            else:
                target_out = None
                target_logits = None
                domain_loss = source_features.sum() * 0.0

            cls_loss = seg_criterion(source_logits, source_mask.long())
            dice_loss = _dice_loss(
                source_logits,
                source_mask.long(),
                ignore_index=ignore_index,
                class_weights=class_weights,
            ) if dice_weight > 0.0 else source_features.sum() * 0.0
            ft_loss = _focal_tversky_loss(
                source_logits,
                source_mask.long(),
                alpha=ft_alpha,
                beta=ft_beta,
                gamma=ft_gamma,
                ignore_index=ignore_index,
                class_weights=class_weights,
            ) if ft_weight > 0.0 else source_features.sum() * 0.0
            sep_loss = seg_criterion(source_out["sep_logits"], source_mask.long())

            if target_on_source_out is not None:
                target_source_logits = target_on_source_out["fusion_logits"]
                target_source_cls_loss = seg_criterion(target_source_logits, source_mask.long())
                target_source_dice_loss = _dice_loss(
                    target_source_logits,
                    source_mask.long(),
                    ignore_index=ignore_index,
                    class_weights=class_weights,
                ) if dice_weight > 0.0 else source_features.sum() * 0.0
                target_source_ft_loss = _focal_tversky_loss(
                    target_source_logits,
                    source_mask.long(),
                    alpha=ft_alpha,
                    beta=ft_beta,
                    gamma=ft_gamma,
                    ignore_index=ignore_index,
                    class_weights=class_weights,
                ) if ft_weight > 0.0 else source_features.sum() * 0.0
                target_source_sep_loss = seg_criterion(target_on_source_out["sep_logits"], source_mask.long())
                target_source_sup_loss = (
                    target_source_cls_loss
                    + dice_weight * target_source_dice_loss
                    + ft_weight * target_source_ft_loss
                    + cfg["train"]["sep_loss_weight"] * target_source_sep_loss
                )
            else:
                target_source_sup_loss = source_features.sum() * 0.0

            gate_s = source_out["gate"].mean(dim=0)
            if enable_uda:
                gate_t = target_out["gate"].mean(dim=0)
                sparse_loss = gate_s.mean() + gate_t.mean()
                excl_loss = (gate_s * gate_t).mean()
            else:
                sparse_loss = gate_s.mean()
                excl_loss = gate_s.mean() * 0.0

            sep_weight = cfg["train"]["sep_loss_weight"]
            alpha = cfg["train"]["gate_sparse_alpha"]
            beta = cfg["train"]["gate_excl_beta"]
            cc_min_pixels = cfg["train"].get("cc_min_pixels", 64)
            s3_target_sup_weight = float(cfg["train"].get("s3_target_sup_weight", 0.0))
            s3_target_sup_loss = source_features.sum() * 0.0

            if enable_uda:
                if s3_lambda > 0.0 and cfg["train"].get("s3_align_mode", "pseudo") == "target_labeled":
                    if target_labeled_loader is None:
                        raise RuntimeError("s3_align_mode='target_labeled' requires data.target_labeled_s3 config")
                    (target_labeled_data, target_labeled_mask), target_labeled_iter = _next_or_restart(
                        target_labeled_iter,
                        target_labeled_loader,
                    )
                    target_labeled_data = torch.nan_to_num(
                        target_labeled_data.to(device),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    target_labeled_mask = torch.nan_to_num(
                        target_labeled_mask.float().to(device),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).long()
                    target_labeled_out = feature_encoder(target_labeled_data, domain="target")
                    if s3_target_sup_weight > 0.0:
                        target_labeled_logits = target_labeled_out["fusion_logits"]
                        target_labeled_cls_loss = seg_criterion(target_labeled_logits, target_labeled_mask.long())
                        target_labeled_dice_loss = _dice_loss(
                            target_labeled_logits,
                            target_labeled_mask.long(),
                            ignore_index=ignore_index,
                            class_weights=class_weights,
                        ) if dice_weight > 0.0 else source_features.sum() * 0.0
                        target_labeled_ft_loss = _focal_tversky_loss(
                            target_labeled_logits,
                            target_labeled_mask.long(),
                            alpha=ft_alpha,
                            beta=ft_beta,
                            gamma=ft_gamma,
                            ignore_index=ignore_index,
                            class_weights=class_weights,
                        ) if ft_weight > 0.0 else source_features.sum() * 0.0
                        target_labeled_sep_loss = seg_criterion(target_labeled_out["sep_logits"], target_labeled_mask.long())
                        s3_target_sup_loss = (
                            target_labeled_cls_loss
                            + dice_weight * target_labeled_dice_loss
                            + ft_weight * target_labeled_ft_loss
                            + sep_weight * target_labeled_sep_loss
                        )
                    cc_align_loss, cc_matched_classes, pseudo_cov, pseudo_fg_ratio, cc_gate_on = _class_conditional_labeled_align_loss(
                        source_feat=source_out["e1_feat"],
                        source_mask=source_mask.long(),
                        target_feat=target_labeled_out["e1_feat"],
                        target_mask=target_labeled_mask.long(),
                        class_num=class_num,
                        min_pixels=cc_min_pixels,
                        ignore_index=ignore_index,
                    )
                else:
                    cc_align_loss, cc_matched_classes, pseudo_cov, pseudo_fg_ratio, cc_gate_on = _class_conditional_align_loss(
                        source_feat=source_out["e1_feat"],
                        source_mask=source_mask.long(),
                        target_feat=target_out["e1_feat"],
                        target_logits=target_logits,
                        class_num=class_num,
                        conf_threshold=pseudo_thr,
                        min_pixels=cc_min_pixels,
                        ignore_index=ignore_index,
                    )
                cc_weight_effective = s3_lambda if cc_gate_on else 0.0
            else:
                cc_align_loss = source_features.sum() * 0.0
                cc_matched_classes = 0
                pseudo_cov = 0.0
                pseudo_fg_ratio = 0.0
                cc_gate_on = False
                cc_weight_effective = 0.0

            total_loss = (
                cls_loss
                + dice_weight * dice_loss
                + ft_weight * ft_loss
                + sep_weight * sep_loss
                + target_source_sup_weight * target_source_sup_loss
                + s3_target_sup_weight * s3_target_sup_loss
                + effective_da_lambda * domain_loss
                + cc_weight_effective * cc_align_loss
                + alpha * sparse_loss
                + beta * excl_loss
            )

            if not torch.isfinite(total_loss):
                nan_skip_count += 1
                if nan_skip_count <= 10 or nan_skip_count % 100 == 0:
                    print(
                        "warning: non-finite loss at step {:>6d} (skip #{}) | cls {} | dice {} | ft {} | sep {} | tgt_sup {} | s3_tgt_sup {} | domain {} | cc {}".format(
                            global_step,
                            nan_skip_count,
                            cls_loss.detach().item() if torch.isfinite(cls_loss) else float("nan"),
                            dice_loss.detach().item() if torch.isfinite(dice_loss) else float("nan"),
                            ft_loss.detach().item() if torch.isfinite(ft_loss) else float("nan"),
                            sep_loss.detach().item() if torch.isfinite(sep_loss) else float("nan"),
                            target_source_sup_loss.detach().item() if torch.isfinite(target_source_sup_loss) else float("nan"),
                            s3_target_sup_loss.detach().item() if torch.isfinite(s3_target_sup_loss) else float("nan"),
                            domain_loss.detach().item() if torch.isfinite(domain_loss) else float("nan"),
                            cc_align_loss.detach().item() if torch.isfinite(cc_align_loss) else float("nan"),
                        )
                    )
                continue

            feature_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), max_norm=5.0)
            feature_optimizer.step()
            domain_optimizer.step()

            epoch_sums["cls"] += cls_loss.item()
            epoch_sums["dice"] += dice_loss.item()
            epoch_sums["ft"] += ft_loss.item()
            epoch_sums["sep"] += sep_loss.item()
            epoch_sums["tgt_sup"] += target_source_sup_loss.item()
            epoch_sums["s3_tgt_sup"] += s3_target_sup_loss.item()
            epoch_sums["domain"] += domain_loss.item()
            epoch_sums["cc"] += cc_align_loss.item()
            epoch_sums["sparse"] += sparse_loss.item()
            epoch_sums["excl"] += excl_loss.item()
            epoch_sums["total"] += total_loss.item()
            epoch_sums["cc_cls"] += float(cc_matched_classes)
            epoch_sums["pseudo_cov"] += float(pseudo_cov)
            epoch_sums["pseudo_fg_ratio"] += float(pseudo_fg_ratio)
            epoch_sums["cc_gate_on"] += 1.0 if cc_gate_on else 0.0
            epoch_sums["cc_w_eff"] += float(cc_weight_effective)
            epoch_updates += 1

        denom = max(1, epoch_updates)
        feature_encoder.eval()

        if target_metric_loader is not None:
            target_val_m_f1, target_val_m_iou, target_val_conf = evaluate_fusion_classifier(
                model=feature_encoder,
                dataloader=target_metric_loader,
                num_classes=class_num,
                device=device,
                domain="target",
            )
            target_metrics = metrics_from_confusion(target_val_conf)
            target_val_oa = target_metrics["oa"]
            save_metric_name = target_metric_name
            save_m_iou = target_val_m_iou
            save_m_f1 = target_val_m_f1
            save_oa = target_val_oa
        else:
            raise RuntimeError("Target validation loader is required for checkpoint selection.")

        miou_history.append(target_val_m_iou)

        if s3_lambda > 0.0:
            s3_active_epochs += 1

        ckpt_dir = cfg["paths"]["checkpoints_dir"]
        ckpt_stage = stage_tag if stage_tag in stage_seen else "S1"
        stage_seen[ckpt_stage] += 1
        stage_ckpt_dir = os.path.join(ckpt_dir, ckpt_stage)
        os.makedirs(stage_ckpt_dir, exist_ok=True)

        current_stage_metrics = {
            "miou": save_m_iou,
            "mf1": save_m_f1,
            "oa": save_oa,
        }
        for metric_key, metric_value in current_stage_metrics.items():
            record = stage_best[ckpt_stage][metric_key]
            if metric_value > record["value"]:
                record["value"] = metric_value
                record["epoch"] = epoch + 1
                torch.save(feature_encoder.state_dict(), os.path.join(stage_ckpt_dir, record["filename"]))
        torch.save(feature_encoder.state_dict(), os.path.join(stage_ckpt_dir, "UP1_feature_encoder_final_0.pkl"))

        feature_encoder.train()
        domain_classifier.train()
        target_log = ""
        if target_metric_loader is not None:
            target_log = (
                " | {}_mF1 {:6.4f} | {}_mIoU {:6.4f} | {}_OA {:6.4f}"
            ).format(
                target_metric_name,
                target_val_m_f1,
                target_metric_name,
                target_val_m_iou,
                target_metric_name,
                target_val_oa,
            )
        print(
            "epoch {:>4d}/{} [{}] | cls {:6.4f} | dice {:6.4f} | ft {:6.4f} | sep {:6.4f} | tgt_sup {:6.4f} | s3_tgt_sup {:6.4f} | domain {:6.4f} | cc {:6.4f} | sparse {:6.4f} | excl {:6.4f} | total {:6.4f} | tgt_sup_w {:5.3f} | s3_tgt_sup_w {:5.3f} | da_w {:5.3f} | cc_w {:5.3f} | cc_w_eff {:5.3f} | save_by {} | thr {:4.2f} | cc_cls {:4.2f} | pseudo_cov {:5.3f} | pseudo_valid {:5.4f} | cc_active {:4.2f}{} | updates {}/{}".format(
                epoch + 1,
                max_epoch,
                stage_tag,
                epoch_sums["cls"] / denom,
                epoch_sums["dice"] / denom,
                epoch_sums["ft"] / denom,
                epoch_sums["sep"] / denom,
                epoch_sums["tgt_sup"] / denom,
                epoch_sums["s3_tgt_sup"] / denom,
                epoch_sums["domain"] / denom,
                epoch_sums["cc"] / denom,
                epoch_sums["sparse"] / denom,
                epoch_sums["excl"] / denom,
                epoch_sums["total"] / denom,
                target_source_sup_weight,
                s3_target_sup_weight if s3_lambda > 0.0 else 0.0,
                effective_da_lambda,
                s3_lambda,
                epoch_sums["cc_w_eff"] / denom,
                save_metric_name,
                pseudo_thr,
                epoch_sums["cc_cls"] / denom,
                epoch_sums["pseudo_cov"] / denom,
                epoch_sums["pseudo_fg_ratio"] / denom,
                epoch_sums["cc_gate_on"] / denom,
                target_log,
                epoch_updates,
                source_batches,
            )
        )

    train_end = time.time()
    print("Training finished!")

    torch.save(
        feature_encoder.state_dict(),
        os.path.join(cfg["paths"]["checkpoints_dir"], "UP1_feature_encoder_final_0.pkl"),
    )
    print("Best checkpoints by stage (saved by {}):".format(target_metric_name))
    for stage in stage_names:
        if stage_seen[stage] <= 0:
            print("  {}: no epoch saved".format(stage))
            continue
        print(
            "  {} | mIoU: {:.4f} (epoch {}) | mF1: {:.4f} (epoch {}) | OA: {:.4f} (epoch {})".format(
                stage,
                stage_best[stage]["miou"]["value"],
                stage_best[stage]["miou"]["epoch"],
                stage_best[stage]["mf1"]["value"],
                stage_best[stage]["mf1"]["epoch"],
                stage_best[stage]["oa"]["value"],
                stage_best[stage]["oa"]["epoch"],
            )
        )
    best_metric = str(cfg["train"].get("best_model_metric", "miou")).lower()
    if best_metric in ("miou", "m_iou", "train_miou"):
        best_ckpt_name = "best_by_miou.pkl"
    elif best_metric in ("mf1", "m_f1", "train_mf1"):
        best_ckpt_name = "best_by_mf1.pkl"
    elif best_metric in ("oa", "acc", "accuracy"):
        best_ckpt_name = "best_by_oa.pkl"
    else:
        best_ckpt_name = "best_by_miou.pkl"
        print(f"[Predict] unknown best_model_metric={best_metric}, fallback to miou")

    target_predict_image_dir = cfg["eval"].get("predict_image_dir", None)
    predict_save_root = cfg["paths"].get("predict_dir", None)
    if predict_save_root:
        for stage in stage_names:
            stage_dir = os.path.join(cfg["paths"]["checkpoints_dir"], stage)
            for metric_key in ("miou", "mf1", "oa"):
                ckpt_name = stage_best[stage][metric_key]["filename"]
                ckpt_path = os.path.join(stage_dir, ckpt_name)
                if not os.path.exists(ckpt_path):
                    continue
                state_dict = torch.load(ckpt_path, map_location=device)
                feature_encoder.load_state_dict(state_dict)
                feature_encoder.to(device)
                feature_encoder.eval()
                if target_metric_loader is not None:
                    _evaluate_named_loader(
                        split_name="Target Val][{}][{}".format(stage, os.path.splitext(ckpt_name)[0]),
                        feature_encoder=feature_encoder,
                        dataloader=target_metric_loader,
                        class_num=class_num,
                        device=device,
                        domain="target",
                    )
                if target_test_loader is not None:
                    _evaluate_named_loader(
                        split_name="Target Test][{}][{}".format(stage, os.path.splitext(ckpt_name)[0]),
                        feature_encoder=feature_encoder,
                        dataloader=target_test_loader,
                        class_num=class_num,
                        device=device,
                        domain="target",
                    )

        predicted_any_stage = False
        predict_ckpt_name = best_ckpt_name
        for stage in stage_names:
            best_ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], stage, predict_ckpt_name)
            if not os.path.exists(best_ckpt_path):
                print(f"[Predict][{stage}] checkpoint not found: {predict_ckpt_name}")
                continue

            predicted_any_stage = True
            state_dict = torch.load(best_ckpt_path, map_location=device)
            feature_encoder.load_state_dict(state_dict)
            feature_encoder.to(device)
            feature_encoder.eval()

            if target_predict_image_dir:
                target_save_dir = os.path.join(predict_save_root, stage, "target")
                target_results = predict_multispectral_images_batch(
                    image_dir=target_predict_image_dir,
                    feature_encoder=feature_encoder,
                    expected_bands=data_cfg["input_bands"],
                    save_dir=target_save_dir,
                    suffix="",
                    domain="target",
                    normalize_mode="auto",
                    output_label_offset=1,
                )
                print(
                    "[Predict][{}][target] using checkpoint: {} | images: {} | saved_dir: {}".format(
                        stage,
                        predict_ckpt_name,
                        len(target_results),
                        target_save_dir,
                    )
                )

        if not predicted_any_stage:
            print(f"[Predict] {predict_ckpt_name} checkpoint not found in S1/S2/S3")
    print("train time per DataSet(s): {:.5f}".format(train_end - train_start))
    return {"accuracy": None, "kappa": None, "confusion_matrix": None}



