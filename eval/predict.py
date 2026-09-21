import os

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

try:
    import rasterio
except Exception:
    rasterio = None


def _ensure_bands(image, expected_bands):
    if image.ndim == 2:
        image = image[:, :, None]
    _, _, channels = image.shape

    if channels == expected_bands:
        return image
    if channels == 3 and expected_bands > 3:
        return np.repeat(image, expected_bands // 3 + 1, axis=2)[:, :, :expected_bands]

    raise ValueError(f"Unsupported channel number {channels}, expect {expected_bands}")


def _read_multiband_image(path):
    if rasterio is not None and str(path).lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as ds:
            arr = ds.read().astype(np.float32)  # [C,H,W]
        arr = np.transpose(arr, (1, 2, 0))  # [H,W,C]
    else:
        arr = imageio.imread(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _normalize_image(image, normalize_mode="none"):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize_mode in ("auto", "imagenet", "255") and image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def _save_mask_with_geo(ref_image_path, save_path, mask):
    mask = np.asarray(mask, dtype=np.uint8)
    is_tif = str(save_path).lower().endswith((".tif", ".tiff"))

    if is_tif:
        if rasterio is None:
            raise RuntimeError("rasterio is required to save georeferenced TIFF predictions")
        with rasterio.open(ref_image_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                dtype=rasterio.uint8,
                count=1,
                height=mask.shape[0],
                width=mask.shape[1],
                transform=src.transform,
                crs=src.crs,
                compress=profile.get("compress") or "lzw",
                nodata=None,
            )
            if src.crs is None:
                raise RuntimeError(f"Reference image has no CRS, cannot save georeferenced mask: {ref_image_path}")
            with rasterio.open(save_path, "w", **profile) as dst:
                dst.write(mask, 1)
        return

    imageio.imwrite(save_path, mask)


def predict_multispectral_images_batch(
    image_dir,
    feature_encoder,
    expected_bands=10,
    save_dir=None,
    suffix="_pred",
    domain="target",
    normalize_mode="none",
    output_label_offset=1,
):
    """
    Full-image logits prediction.

    Args:
        image_dir: input image folder
        feature_encoder: trained model
        expected_bands: expected channel count
        save_dir: if provided, save predicted masks as .tif into this folder
        suffix: suffix added to output filename stem, pass "" to keep original stem
        domain: "source" or "target"
        normalize_mode: kept for compatibility; images are not rescaled

    Returns:
        dict: {image_name: predicted_mask_uint8}
    """
    device = next(feature_encoder.parameters()).device
    feature_encoder.eval()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    image_files = sorted(
        [name for name in os.listdir(image_dir) if name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    )

    results = {}
    with torch.no_grad():
        for image_name in tqdm(image_files, desc="[Predict] Images", unit="img"):
            image_path = os.path.join(image_dir, image_name)
            image = _read_multiband_image(image_path)
            image = _normalize_image(image, normalize_mode=normalize_mode)
            image = _ensure_bands(image, expected_bands)

            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            _, logits = feature_encoder.forward_fusion(image_tensor, domain=domain)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            save_preds = (preds + int(output_label_offset)).astype(np.uint8)
            results[image_name] = save_preds

            if save_dir is not None:
                stem = os.path.splitext(image_name)[0]
                save_path = os.path.join(save_dir, f"{stem}{suffix}.png")
                imageio.imwrite(save_path, save_preds)

    return results
