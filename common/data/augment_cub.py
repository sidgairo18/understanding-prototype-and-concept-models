"""Generate a ProtoPNet-style OFFLINE augmented, bbox-cropped CUB *training* set.

The paper trains on a large offline-augmented set (each cropped bird expanded ~30-40×). This
script reproduces that idea: for every **training** image it crops to the bird's bounding box,
then writes the cropped original plus ``per_image`` random-augmented variants (rotation /
shear / perspective / flip) into a torchvision **ImageFolder** layout::

    <out>/<class_dir>/<image_stem>_aug{NN}.jpg

Only the *train* split is expanded. Test and prototype-**push** loaders keep using the
un-augmented cropped CUB (``common.data.cub.CUB200(crop_to_bbox=True)``), matching the paper:
train on the augmented set, but project prototypes onto the original cropped patches.

Class-dir names are the original CUB names ("001.Black_footed_Albatross", ...), so their
alphabetical order (what ImageFolder uses for labels) matches CUB200's class-id order — the
augmented train labels line up with the CUB200 test labels.

Run (from repo root)::

    python -m common.data.augment_cub --cub-root $CUB_ROOT \
        --out /path/to/cub200_crop_aug_train --num-classes 200 --per-image 15 --workers 16
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

from PIL import Image
from torchvision import transforms

from common.data.cub import resolve_root


# ------------------------------------------------------------------ metadata readers
def _read_kv(root: str, filename: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(os.path.join(root, filename)) as f:
        for line in f:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
    return out


def _read_bboxes(root: str) -> dict[str, tuple[float, float, float, float]]:
    out: dict[str, tuple[float, float, float, float]] = {}
    with open(os.path.join(root, "bounding_boxes.txt")) as f:
        for line in f:
            image_id, x, y, w, h = line.split()
            out[image_id] = (float(x), float(y), float(w), float(h))
    return out


# ------------------------------------------------------------------ per-image worker
def _aug_pipeline(img_size: int) -> transforms.Compose:
    """Geometric augmentation producing a PIL image (fresh random params per call)."""
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
    ])


def _process_one(task) -> int:
    path, bbox, out_class_dir, stem, per_image, img_size = task
    try:
        img = Image.open(path).convert("RGB")
        if bbox is not None:
            x, y, w, h = bbox
            img = img.crop((x, y, x + w, y + h))     # crop to the bird
        os.makedirs(out_class_dir, exist_ok=True)
        # aug00 = the plain cropped original (resized), then per_image augmented variants.
        img.resize((img_size, img_size)).save(
            os.path.join(out_class_dir, f"{stem}_aug00.jpg"), quality=95)
        aug = _aug_pipeline(img_size)
        for n in range(1, per_image + 1):
            aug(img).save(os.path.join(out_class_dir, f"{stem}_aug{n:02d}.jpg"), quality=95)
        return per_image + 1
    except Exception as e:  # keep going; report the failure count at the end
        print(f"[warn] failed {path}: {e}")
        return 0


# ------------------------------------------------------------------ driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cub-root", default=None, help="CUB_200_2011 dir (or CUB_ROOT env var).")
    ap.add_argument("--out", required=True, help="Output ImageFolder root for the augmented train set.")
    ap.add_argument("--num-classes", type=int, default=200)
    ap.add_argument("--per-image", type=int, default=15, help="Augmented variants per training image.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = resolve_root(args.cub_root)
    images = _read_kv(root, "images.txt")
    labels = _read_kv(root, "image_class_labels.txt")
    splits = _read_kv(root, "train_test_split.txt")
    bboxes = _read_bboxes(root)

    kept = set(sorted({int(c) for c in labels.values()})[: args.num_classes])
    tasks = []
    for image_id, rel in images.items():
        if int(labels[image_id]) not in kept or splits[image_id] != "1":
            continue                                 # kept classes, train split only
        class_dir = rel.split("/")[0]                # e.g. "001.Black_footed_Albatross"
        stem = os.path.splitext(os.path.basename(rel))[0]
        tasks.append((os.path.join(root, "images", rel), bboxes.get(image_id),
                      os.path.join(args.out, class_dir), stem, args.per_image, args.img_size))

    total = len(tasks) * (args.per_image + 1)
    print(f"generating ~{total} images ({len(tasks)} originals × {args.per_image + 1}) "
          f"for {len(kept)} classes -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    with Pool(args.workers) as pool:
        counts = pool.map(_process_one, tasks, chunksize=8)
    print(f"done: wrote {sum(counts)} images across "
          f"{len({t[2] for t in tasks})} class dirs under {args.out}")


if __name__ == "__main__":
    main()
