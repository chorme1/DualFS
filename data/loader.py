import os
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import (
    SegmentationImageDataset,
    SegmentationPatchDataset,
    UnlabeledImageDataset,
    UnlabeledPatchDataset,
)


class ForegroundAwareBatchSampler(torch.utils.data.Sampler):
    def __init__(self, fg_indices, bg_indices, batch_size, fg_per_batch, num_batches):
        self.fg_indices = list(fg_indices)
        self.bg_indices = list(bg_indices)
        self.batch_size = int(batch_size)
        self.fg_per_batch = int(max(0, min(fg_per_batch, batch_size)))
        self.num_batches = int(num_batches)

    def __len__(self):
        return self.num_batches

    def _take(self, pool, count, pointer):
        if count <= 0 or len(pool) == 0:
            return [], pointer
        out = []
        for _ in range(count):
            if pointer >= len(pool):
                np.random.shuffle(pool)
                pointer = 0
            out.append(pool[pointer])
            pointer += 1
        return out, pointer

    def __iter__(self):
        fg_pool = self.fg_indices.copy()
        bg_pool = self.bg_indices.copy()
        np.random.shuffle(fg_pool)
        np.random.shuffle(bg_pool)
        fg_ptr, bg_ptr = 0, 0

        for _ in range(self.num_batches):
            if len(fg_pool) == 0 and len(bg_pool) == 0:
                break
            use_fg = self.fg_per_batch if len(fg_pool) > 0 else 0
            use_bg = self.batch_size - use_fg
            if len(bg_pool) == 0:
                use_fg = self.batch_size
                use_bg = 0

            batch_fg, fg_ptr = self._take(fg_pool, use_fg, fg_ptr)
            batch_bg, bg_ptr = self._take(bg_pool, use_bg, bg_ptr)
            batch = batch_fg + batch_bg

            if len(batch) < self.batch_size:
                extra = self.batch_size - len(batch)
                fill_pool = fg_pool if len(fg_pool) > 0 else bg_pool
                fill_ptr = fg_ptr if len(fg_pool) > 0 else bg_ptr
                fill, fill_ptr = self._take(fill_pool, extra, fill_ptr)
                batch += fill
                if len(fg_pool) > 0:
                    fg_ptr = fill_ptr
                else:
                    bg_ptr = fill_ptr

            np.random.shuffle(batch)
            yield batch


def _foreground_indices(mask_paths, mask_reader, foreground_class_index=1, min_fg_pixels=1):
    fg_indices = []
    bg_indices = []
    for idx, mask_path in enumerate(mask_paths):
        mask = mask_reader(mask_path)
        fg_pixels = int((mask == foreground_class_index).sum())
        if fg_pixels >= min_fg_pixels:
            fg_indices.append(idx)
        else:
            bg_indices.append(idx)
    return fg_indices, bg_indices


def _ensure_bands(image: np.ndarray, expected_bands: int) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    _, _, channels = image.shape

    if channels == expected_bands:
        return image

    if channels == 3 and expected_bands > 3:
        image = np.repeat(image, expected_bands // 3 + 1, axis=2)[:, :, :expected_bands]
        return image

    raise ValueError(f"Unsupported channel number {channels}, expected {expected_bands}")


def _read_image(path: str, normalize: bool, expected_bands: int) -> np.ndarray:
    image = imageio.imread(path).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = _ensure_bands(image, expected_bands)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize and image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def _read_mask(path: str, class_num: int) -> np.ndarray:
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
    # LoveDA labels: 0 is nodata/ignore, 1-7 are valid classes.
    out = np.full_like(mask, fill_value=255, dtype=np.int64)
    valid = (mask >= 1) & (mask <= class_num)
    out[valid] = mask[valid] - 1
    return out


def _list_image_files(image_dir: str, limit_images=None) -> List[str]:
    files = [
        file_name
        for file_name in sorted(os.listdir(image_dir))
        if file_name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    ]
    if limit_images is not None:
        files = files[:limit_images]
    return files


def _collect_labeled_pairs(
    image_dir: str,
    mask_dir: str,
    class_num: int,
    limit_images=None,
) -> Tuple[List[str], List[str]]:
    image_files = _list_image_files(image_dir, limit_images=limit_images)
    image_paths: List[str] = []
    mask_paths: List[str] = []

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)
        if not os.path.exists(mask_path):
            stem, _ = os.path.splitext(image_name)
            candidates = [
                os.path.join(mask_dir, stem + ".tif"),
                os.path.join(mask_dir, stem + ".tiff"),
                os.path.join(mask_dir, stem + ".png"),
            ]
            mask_path = next((path for path in candidates if os.path.exists(path)), None)
        if mask_path is None or not os.path.exists(mask_path):
            continue
        image_paths.append(image_path)
        mask_paths.append(mask_path)

    if not image_paths:
        raise RuntimeError(f"No matched image-mask pairs found in {image_dir} and {mask_dir}")

    class_presence = np.zeros(class_num, dtype=np.int64)
    for mask_path in mask_paths:
        mask = _read_mask(mask_path, class_num)
        for class_index in range(class_num):
            if np.any(mask == class_index):
                class_presence[class_index] += 1
    missing_classes = [index for index, count in enumerate(class_presence) if count == 0]
    if missing_classes:
        raise RuntimeError(
            "Some classes are missing in labeled source images: "
            f"missing_classes={missing_classes}, class_presence={class_presence.tolist()}"
        )

    return image_paths, mask_paths


def select_labeled_pairs(image_paths, mask_paths, num_labeled=None, seed=0):
    pairs = list(zip(image_paths, mask_paths))
    if num_labeled is None:
        return image_paths, mask_paths

    num_labeled = int(num_labeled)
    if num_labeled <= 0:
        raise RuntimeError("num_labeled must be positive.")
    if num_labeled >= len(pairs):
        return image_paths, mask_paths

    rng = np.random.RandomState(int(seed))
    indices = np.sort(rng.choice(len(pairs), size=num_labeled, replace=False))
    selected = [pairs[int(idx)] for idx in indices.tolist()]
    return [pair[0] for pair in selected], [pair[1] for pair in selected]


def _valid_centers(mask, half):
    h, w = mask.shape
    valid = np.zeros_like(mask, dtype=bool)
    valid[half:h - half, half:w - half] = True
    return valid


def _sample_coords(coords, count, rng=None):
    if count <= 0 or coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    replace = coords.shape[0] < count
    chooser = rng if rng is not None else np.random
    idx = chooser.choice(coords.shape[0], size=count, replace=replace)
    return coords[idx]


def _patch_quality_ok(mask_patch, foreground_class_index=1, min_fg_pixels=1, min_valid_ratio=0.0, ignore_index=2):
    fg_pixels = int((mask_patch == foreground_class_index).sum())
    if ignore_index is None:
        valid_ratio = 1.0
    else:
        valid_ratio = float((mask_patch != ignore_index).mean())
    return fg_pixels >= int(min_fg_pixels) and valid_ratio >= float(min_valid_ratio)


def _sample_patches_with_quality(
    image,
    mask,
    coords,
    need_count,
    half,
    foreground_class_index=1,
    min_fg_pixels=1,
    min_valid_ratio=0.0,
    ignore_index=2,
    rng=None,
    max_retry_factor=30,
):
    if need_count <= 0 or coords.shape[0] == 0:
        return []

    chooser = rng if rng is not None else np.random
    max_attempts = max(int(need_count) * int(max_retry_factor), int(max_retry_factor))

    accepted = []
    attempts = 0
    while len(accepted) < int(need_count) and attempts < max_attempts:
        idx = int(chooser.choice(coords.shape[0]))
        r, c = coords[idx]
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - half, c + half + 1
        image_patch = image[r0:r1, c0:c1, :]
        mask_patch = mask[r0:r1, c0:c1]
        if _patch_quality_ok(
            mask_patch,
            foreground_class_index=foreground_class_index,
            min_fg_pixels=min_fg_pixels,
            min_valid_ratio=min_valid_ratio,
            ignore_index=ignore_index,
        ):
            accepted.append((image_patch, mask_patch))
        attempts += 1

    return accepted


def _extract_labeled_patches(
    image_paths,
    mask_paths,
    image_reader,
    mask_reader,
    patch_size,
    patches_per_image,
    fg_patch_ratio,
    foreground_class_index=1,
    min_fg_pixels_per_patch=1,
    min_valid_ratio=0.0,
    ignore_index=2,
    rng=None,
):
    half = patch_size // 2
    patch_images = []
    patch_masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        image = image_reader(img_path)
        mask = mask_reader(mask_path)
        valid = _valid_centers(mask, half)

        fg_coords = np.argwhere((mask == foreground_class_index) & valid)
        bg_coords = np.argwhere((mask == 0) & valid)
        if bg_coords.shape[0] == 0 and fg_coords.shape[0] == 0:
            continue

        fg_count = int(round(patches_per_image * fg_patch_ratio))
        bg_count = max(0, patches_per_image - fg_count)
        if fg_coords.shape[0] == 0:
            fg_count = 0
            bg_count = patches_per_image

        fg_samples = _sample_patches_with_quality(
            image=image,
            mask=mask,
            coords=fg_coords,
            need_count=fg_count,
            half=half,
            foreground_class_index=foreground_class_index,
            min_fg_pixels=min_fg_pixels_per_patch,
            min_valid_ratio=min_valid_ratio,
            ignore_index=ignore_index,
            rng=rng,
        )
        bg_samples = _sample_patches_with_quality(
            image=image,
            mask=mask,
            coords=bg_coords,
            need_count=bg_count,
            half=half,
            foreground_class_index=foreground_class_index,
            min_fg_pixels=0,
            min_valid_ratio=min_valid_ratio,
            ignore_index=ignore_index,
            rng=rng,
        )

        sample_pairs = fg_samples + bg_samples
        if len(sample_pairs) == 0:
            continue
        if rng is None:
            np.random.shuffle(sample_pairs)
        else:
            rng.shuffle(sample_pairs)

        for image_patch, mask_patch in sample_pairs:
            patch_images.append(image_patch)
            patch_masks.append(mask_patch)

    if len(patch_images) == 0:
        raise RuntimeError(
            "No labeled patches extracted after quality constraints. "
            "Try lowering min_fg_pixels_per_patch or min_valid_ratio."
        )
    return patch_images, patch_masks


def _extract_unlabeled_patches(
    image_paths,
    image_reader,
    patch_size,
    patches_per_image,
):
    half = patch_size // 2
    patches = []
    for img_path in image_paths:
        image = image_reader(img_path)
        h, w, _ = image.shape
        if h <= 2 * half or w <= 2 * half:
            continue
        rs = np.random.randint(half, h - half, size=patches_per_image)
        cs = np.random.randint(half, w - half, size=patches_per_image)
        for r, c in zip(rs, cs):
            r0, r1 = r - half, r + half + 1
            c0, c1 = c - half, c + half + 1
            patches.append(image[r0:r1, c0:c1, :])
    if len(patches) == 0:
        raise RuntimeError("No unlabeled patches extracted. Check patch_size and image sizes.")
    return patches


def get_train_test_loader_multi(
    image_dir,
    mask_dir,
    class_num,
    limit_images=5,
    normalize=True,
    expected_bands=10,
    train_all_samples=True,
    train_batch_size=1,
    test_batch_size=1,
    foreground_aware_sampling=False,
    foreground_class_index=1,
    fg_min_pixels_per_image=1,
    fg_per_batch=1,
    use_patch_training=False,
    patch_size=21,
    source_patches_per_image=128,
    source_eval_patches_per_image=128,
    fg_patch_ratio=0.75,
    eval_patch_seed=0,
    min_fg_pixels_per_patch=1,
    min_valid_ratio=0.0,
    ignore_index=2,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=limit_images,
    )

    # Source images are always used as the training pool. Validation should come
    # from the independent cfg["data"]["val"] dataset, not an internal 80/20 split.
    train_image_paths = image_paths
    train_mask_paths = mask_paths
    test_image_paths = image_paths
    test_mask_paths = mask_paths

    image_reader = lambda path: _read_image(path, normalize, expected_bands)
    mask_reader = lambda path: _read_mask(path, class_num)

    if use_patch_training:
        train_patches, train_masks = _extract_labeled_patches(
            train_image_paths,
            train_mask_paths,
            image_reader=image_reader,
            mask_reader=mask_reader,
            patch_size=patch_size,
            patches_per_image=source_patches_per_image,
            fg_patch_ratio=fg_patch_ratio,
            foreground_class_index=foreground_class_index,
            min_fg_pixels_per_patch=min_fg_pixels_per_patch,
            min_valid_ratio=min_valid_ratio,
            ignore_index=ignore_index,
            rng=None,
        )
        eval_rng = np.random.RandomState(eval_patch_seed)
        test_patches, test_masks = _extract_labeled_patches(
            test_image_paths,
            test_mask_paths,
            image_reader=image_reader,
            mask_reader=mask_reader,
            patch_size=patch_size,
            patches_per_image=source_eval_patches_per_image,
            fg_patch_ratio=fg_patch_ratio,
            foreground_class_index=foreground_class_index,
            min_fg_pixels_per_patch=min_fg_pixels_per_patch,
            min_valid_ratio=min_valid_ratio,
            ignore_index=ignore_index,
            rng=eval_rng,
        )
        train_dataset = SegmentationPatchDataset(train_patches, train_masks)
        test_dataset = SegmentationPatchDataset(test_patches, test_masks)
        print(
            "Patch training enabled: patch_size={} source_patches={} test_patches={}".format(
                patch_size, len(train_dataset), len(test_dataset)
            )
        )
    else:
        train_dataset = SegmentationImageDataset(
            train_image_paths,
            train_mask_paths,
            image_reader=image_reader,
            mask_reader=mask_reader,
        )
        test_dataset = SegmentationImageDataset(
            test_image_paths,
            test_mask_paths,
            image_reader=image_reader,
            mask_reader=mask_reader,
        )

    if foreground_aware_sampling and train_batch_size > 1:
        fg_indices, bg_indices = _foreground_indices(
            train_mask_paths,
            mask_reader=lambda p: _read_mask(p, class_num),
            foreground_class_index=foreground_class_index,
            min_fg_pixels=fg_min_pixels_per_image,
        )
        if len(fg_indices) > 0 and len(bg_indices) > 0:
            num_batches = int(np.ceil(len(train_dataset) / float(train_batch_size)))
            batch_sampler = ForegroundAwareBatchSampler(
                fg_indices=fg_indices,
                bg_indices=bg_indices,
                batch_size=train_batch_size,
                fg_per_batch=fg_per_batch,
                num_batches=num_batches,
            )
            print(
                "Foreground-aware sampling enabled: fg_images={} bg_images={} fg_per_batch={}".format(
                    len(fg_indices), len(bg_indices), fg_per_batch
                )
            )
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)
        else:
            raise RuntimeError(
                "Foreground-aware sampling requires both fg and bg groups, but got "
                f"fg_images={len(fg_indices)}, bg_images={len(bg_indices)}. "
                "Adjust fg_min_pixels_per_image/fg_per_batch or disable foreground_aware_sampling."
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader, None


def get_eval_loader(
    image_dir,
    mask_dir,
    class_num,
    limit_images=5,
    normalize=True,
    expected_bands=10,
    batch_size=1,
):
    image_paths, mask_paths = _collect_labeled_pairs(
        image_dir=image_dir,
        mask_dir=mask_dir,
        class_num=class_num,
        limit_images=limit_images,
    )
    eval_dataset = SegmentationImageDataset(
        image_paths,
        mask_paths,
        image_reader=lambda path: _read_image(path, normalize, expected_bands),
        mask_reader=lambda path: _read_mask(path, class_num),
    )
    return DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_unlabeled_loader(
    image_dir,
    limit_images=5,
    normalize=True,
    batch_size=1,
    expected_bands=10,
    use_patch_training=False,
    patch_size=21,
    target_patches_per_image=128,
):
    image_files = _list_image_files(image_dir, limit_images=limit_images)
    image_paths = [os.path.join(image_dir, image_name) for image_name in image_files]
    if not image_paths:
        raise RuntimeError(f"No image files found in target directory: {image_dir}")

    image_reader = lambda path: _read_image(path, normalize, expected_bands)
    if use_patch_training:
        target_patches = _extract_unlabeled_patches(
            image_paths=image_paths,
            image_reader=image_reader,
            patch_size=patch_size,
            patches_per_image=target_patches_per_image,
        )
        target_dataset = UnlabeledPatchDataset(target_patches)
        print("Target patch loader enabled: patch_size={} target_patches={}".format(patch_size, len(target_dataset)))
    else:
        target_dataset = UnlabeledImageDataset(
            image_paths,
            image_reader=image_reader,
        )
    return DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
