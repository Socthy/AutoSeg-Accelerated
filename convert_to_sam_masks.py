"""
Convert AutoSeg-SAM2 final-output (.npy) masks to SegGAussiansAndMesh format.

Compatible with:
  - extract_segment_everything_masks.py  ->  <image_root>/sam_masks/<stem>.pt
  - get_scale.py                         ->  load_masks() from sam_masks/

AutoSeg saves per frame:
  <autoseg_output>/<level>/final-output/mask_XXX.npy
  np.array(mask_list), each mask typically (1, H, W) bool from SAM2 video tracking.

Usage (from this directory):
  python convert_to_sam_masks.py \\
    --image_root /path/to/dataset \\
    --autoseg_output /path/to/output/figurines \\
    --level large
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --- mask I/O (compatible with SegGAussiansAndMesh utils/mask_io.py) ---


def _pack_bool_masks(masks: torch.Tensor) -> Dict[str, Any]:
    if masks.dtype != torch.bool:
        masks = masks.bool()
    if masks.dim() != 3:
        raise ValueError(f"masks must be (N,H,W), got {tuple(masks.shape)}")
    n, h, w = (int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2]))
    flat = masks.reshape(n, h * w).cpu().numpy().astype(np.uint8)
    packed = np.packbits(flat, axis=1)
    return {
        "__format__": "packbits_v1",
        "shape": (n, h, w),
        "packed": torch.from_numpy(packed),
    }


def _unpack_bool_masks(obj: Dict[str, Any]) -> torch.Tensor:
    if obj.get("__format__") != "packbits_v1":
        raise ValueError(f"Unknown packed mask format: {obj.get('__format__')}")
    n, h, w = obj["shape"]
    packed = obj["packed"]
    if isinstance(packed, torch.Tensor):
        packed_np = packed.cpu().numpy()
    else:
        packed_np = np.asarray(packed)
    unpacked = np.unpackbits(packed_np, axis=1)
    hw = int(h) * int(w)
    unpacked = unpacked[:, :hw]
    return torch.from_numpy(unpacked.astype(np.uint8)).view(int(n), int(h), int(w)).bool()


def save_masks(path: str, masks: torch.Tensor, compress: bool = True) -> None:
    """Save (N,H,W) bool masks to .pt."""
    if compress:
        torch.save(_pack_bool_masks(masks), path)
    else:
        torch.save(masks.bool().cpu(), path)


def load_masks(path: str) -> torch.Tensor:
    """Load masks saved by save_masks() or legacy torch.save(bool_tensor)."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj.get("__format__") == "packbits_v1":
        return _unpack_bool_masks(obj)
    t = torch.as_tensor(obj)
    if t.dim() != 3:
        raise ValueError(f"Expected (N,H,W) masks in {path}, got {tuple(t.shape)}")
    return t.bool().cpu()


# --- AutoSeg conversion ---


def frame_sort_key(filename: str):
    stem = os.path.splitext(filename)[0]
    if stem.isdigit():
        return (0, int(stem), "")
    match = re.search(r"(\d+)$", stem)
    if match:
        return (1, stem[: match.start()], int(match.group(1)), stem)
    return (2, stem)


def image_stem(filename: str) -> str:
    """Match extract_segment_everything_masks.py: path.split('.')[0]."""
    return filename.split(".")[0]


def normalize_frame_masks(raw: np.ndarray) -> np.ndarray:
    """(N,H,W) bool from AutoSeg (N,1,H,W) or (N,H,W)."""
    a = np.asarray(raw)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    elif a.ndim == 4:
        if a.shape[1] == 1:
            a = np.squeeze(a, axis=1)
        else:
            raise ValueError(f"Expected (N,1,H,W), got {a.shape}")
    elif a.ndim != 3:
        raise ValueError(f"Unexpected mask ndim={a.ndim} shape={a.shape}")
    return a.astype(bool)


def mask_npy_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("mask_"):
        suffix = stem[5:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, stem)


def list_images(image_dir: str) -> list[str]:
    names = [
        p
        for p in os.listdir(image_dir)
        if os.path.splitext(p)[-1] in IMAGE_EXTENSIONS
    ]
    names.sort(key=frame_sort_key)
    return names


def list_mask_npy(mask_dir: str) -> list[str]:
    paths = [
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".npy")
    ]
    paths.sort(key=mask_npy_sort_key)
    return paths


def masks_to_tensor(masks_uhw: np.ndarray, max_masks: int = 0) -> torch.Tensor:
    """Build (N,H,W) bool tensor; drop empty masks like extract_segment_everything_masks."""
    mask_list = []
    for i in range(masks_uhw.shape[0]):
        m = masks_uhw[i]
        if m.sum() == 0:
            continue
        if len(np.unique(m)) < 2:
            continue
        mask_list.append(torch.from_numpy(m).bool())
    if not mask_list:
        h, w = int(masks_uhw.shape[1]), int(masks_uhw.shape[2])
        return torch.zeros((0, h, w), dtype=torch.bool)
    stacked = torch.stack(mask_list, dim=0)
    if max_masks > 0 and stacked.shape[0] > max_masks:
        areas = stacked.flatten(1).sum(dim=1)
        keep = torch.topk(areas, k=max_masks, largest=True).indices
        stacked = stacked[keep]
    return stacked


def convert(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    compress: bool = True,
    skip_existing: bool = False,
    max_masks_per_image: int = 0,
) -> None:
    image_names = list_images(image_dir)
    mask_paths = list_mask_npy(mask_dir)

    if not image_names:
        raise FileNotFoundError(f"No images under {image_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"No mask_*.npy under {mask_dir}")

    if len(image_names) != len(mask_paths):
        raise ValueError(
            f"Image count ({len(image_names)}) != mask file count ({len(mask_paths)}).\n"
            f"  images: {image_dir}\n"
            f"  masks:  {mask_dir}\n"
            "Ensure AutoSeg --video_path matches this image folder and frame order."
        )

    os.makedirs(output_dir, exist_ok=True)

    for img_name, mask_path in tqdm(
        zip(image_names, mask_paths), total=len(image_names), desc="convert"
    ):
        stem = image_stem(img_name)
        out_path = os.path.join(output_dir, stem + ".pt")
        if skip_existing and os.path.isfile(out_path):
            continue

        raw = np.load(mask_path)
        masks_uhw = normalize_frame_masks(raw)
        masks_t = masks_to_tensor(masks_uhw, max_masks=max_masks_per_image)
        save_masks(out_path, masks_t, compress=compress)

    print(f"Done. Wrote {len(image_names)} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AutoSeg-SAM2 masks to sam_masks/*.pt for SegGAussiansAndMesh"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Dataset root (must contain images/ unless --image_dir is set)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Folder of input frames (default: <image_root>/images)",
    )
    parser.add_argument(
        "--autoseg_output",
        type=str,
        required=True,
        help="AutoSeg --output_dir (contains <level>/final-output/)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="large",
        choices=("default", "small", "middle", "large"),
        help="AutoSeg mask level",
    )
    parser.add_argument(
        "--mask_leaf",
        type=str,
        default="final-output",
        help="Subdir under autoseg_output/<level>/ (default: final-output)",
    )
    parser.add_argument(
        "--sam_masks_dir",
        type=str,
        default=None,
        help="Output dir (default: <image_root>/sam_masks)",
    )
    parser.add_argument(
        "--max_masks_per_image",
        type=int,
        default=0,
        help=">0: keep at most N masks per frame (largest area first)",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="Save raw bool tensor (larger files, still loadable by load_masks)",
    )
    args = parser.parse_args()

    image_root = os.path.abspath(args.image_root)
    image_dir = args.image_dir or os.path.join(image_root, "images")
    image_dir = os.path.abspath(image_dir)

    mask_dir = os.path.join(
        os.path.abspath(args.autoseg_output), args.level, args.mask_leaf
    )
    output_dir = args.sam_masks_dir or os.path.join(image_root, "sam_masks")
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"AutoSeg mask directory not found: {mask_dir}\n"
            "Run auto-mask-fast.py successfully first."
        )

    convert(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        compress=not args.no_compress,
        skip_existing=args.skip_existing,
        max_masks_per_image=args.max_masks_per_image,
    )


if __name__ == "__main__":
    main()
