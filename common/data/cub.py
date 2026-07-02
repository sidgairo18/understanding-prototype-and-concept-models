"""CUB-200-2011 dataset loader shared by all POCs.

CUB-200-2011 (Caltech-UCSD Birds) has 200 fine-grained bird classes and 11,788 images,
with an official train/test split. It is the common benchmark for prototype and
part-discovery methods, so every POC in this repo trains on it.

Why a custom loader (instead of `torchvision.datasets`)?
  * We want a built-in *subset* switch (`num_classes`, `images_per_class`) so the default
    POC run is tiny and fast, and scaling to full CUB is a one-line config change.
  * Part-discovery POCs (SCOPS, PDiscoFormer) optionally need the bounding boxes / a
    foreground crop, which this loader can expose.

Expected on-disk layout (the standard CUB_200_2011 extraction)::

    <data_root>/
        images.txt                # <image_id> <relative/path.jpg>
        image_class_labels.txt    # <image_id> <class_id>           (class_id is 1-indexed)
        train_test_split.txt      # <image_id> <is_training_image>   (1 = train, 0 = test)
        classes.txt               # <class_id> <class_name>
        bounding_boxes.txt        # <image_id> <x> <y> <width> <height>
        images/<class>/<file>.jpg

Point the loader at `<data_root>` via the `CUB_ROOT` env var or the `data_root` argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics — backbones in `common.backbones` are ImageNet-pretrained, so we
# normalize with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_root(data_root: str | None) -> str:
    """Resolve the CUB root from an explicit path or the ``CUB_ROOT`` env var."""
    root = data_root or os.environ.get("CUB_ROOT")
    if not root:
        raise ValueError(
            "CUB root not set. Pass data_root=... or export CUB_ROOT=/path/to/CUB_200_2011"
        )
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CUB root does not exist: {root}")
    return root


def build_transforms(img_size: int = 224, train: bool = True,
                     strong: bool = False) -> transforms.Compose:
    """Standard ImageNet-style transforms used across POCs.

    Individual POCs can override this (e.g. ProtoPNet uses extra augmentation, SCOPS
    uses paired geometric transforms), but this is the sensible shared default.

    ``strong=True`` adds paper-style geometric augmentation for *training* (random
    rotation / shear / perspective). ProtoPNet's recipe augments the (bbox-cropped)
    training set this way; we apply it **online** here rather than pre-expanding ~30× to
    disk — same regularization, no dataset generation step.
    """
    if not train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    ops = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if strong:
        ops += [
            transforms.RandomAffine(degrees=15, shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


@dataclass
class _Sample:
    image_id: int
    path: str          # absolute path to the .jpg
    label: int         # 0-indexed *remapped* class label (within the chosen subset)
    bbox: tuple[float, float, float, float] | None  # (x, y, w, h) or None


class CUB200(Dataset):
    """CUB-200-2011 with an optional class/image subset for fast POC iteration.

    Args:
        data_root: path to the extracted ``CUB_200_2011`` dir (or None to use ``CUB_ROOT``).
        train: use the official training split if True, else the test split.
        transform: a torchvision transform; if None, :func:`build_transforms` is used.
        num_classes: keep only the first N class ids (None = all 200). Default keeps the
            run tiny.
        images_per_class: cap images kept per class (None = all). Further shrinks the run.
        return_bbox: if True, ``__getitem__`` also returns the (x, y, w, h) bbox.
        crop_to_bbox: if True, crop each image to its CUB bounding box before transforming
            (the ProtoPNet paper trains/evaluates on bird-cropped images).

    ``__getitem__`` returns ``(image_tensor, label)`` — or ``(image, label, bbox)`` when
    ``return_bbox`` is set. Labels are remapped to ``0..num_classes-1`` over the kept
    classes so they index a small classifier head directly.
    """

    def __init__(
        self,
        data_root: str | None = None,
        train: bool = True,
        transform=None,
        num_classes: int | None = 10,
        images_per_class: int | None = None,
        return_bbox: bool = False,
        crop_to_bbox: bool = False,
    ) -> None:
        self.root = resolve_root(data_root)
        self.train = train
        self.transform = transform or build_transforms(train=train)
        self.return_bbox = return_bbox
        self.crop_to_bbox = crop_to_bbox

        images = self._read_kv("images.txt")                 # id -> relative path
        labels = self._read_kv("image_class_labels.txt")     # id -> class_id (1-indexed)
        splits = self._read_kv("train_test_split.txt")       # id -> "1"/"0"
        bboxes = self._read_bboxes() if (return_bbox or crop_to_bbox) else {}

        # Choose which (original, 1-indexed) class ids to keep.
        all_class_ids = sorted({int(c) for c in labels.values()})
        kept_class_ids = all_class_ids if num_classes is None else all_class_ids[:num_classes]
        keep_set = set(kept_class_ids)
        # Remap kept class ids -> contiguous 0-indexed labels.
        self.class_id_to_label = {cid: i for i, cid in enumerate(kept_class_ids)}
        self.num_classes = len(kept_class_ids)

        per_class_count: dict[int, int] = {}
        self.samples: list[_Sample] = []
        for image_id, rel_path in images.items():
            class_id = int(labels[image_id])
            if class_id not in keep_set:
                continue
            is_train = splits[image_id] == "1"
            if is_train != train:
                continue
            if images_per_class is not None:
                if per_class_count.get(class_id, 0) >= images_per_class:
                    continue
                per_class_count[class_id] = per_class_count.get(class_id, 0) + 1
            self.samples.append(
                _Sample(
                    image_id=int(image_id),
                    path=os.path.join(self.root, "images", rel_path),
                    label=self.class_id_to_label[class_id],
                    bbox=bboxes.get(image_id),
                )
            )

        if not self.samples:
            raise RuntimeError(
                "No CUB samples selected — check data_root and the subset settings."
            )

    # ------------------------------------------------------------------ helpers
    def _read_kv(self, filename: str) -> dict[str, str]:
        """Read a 'whitespace-separated, first token is image_id' CUB metadata file."""
        path = os.path.join(self.root, filename)
        out: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    out[parts[0]] = parts[1].strip()
        return out

    def _read_bboxes(self) -> dict[str, tuple[float, float, float, float]]:
        path = os.path.join(self.root, "bounding_boxes.txt")
        out: dict[str, tuple[float, float, float, float]] = {}
        with open(path) as f:
            for line in f:
                image_id, x, y, w, h = line.split()
                out[image_id] = (float(x), float(y), float(w), float(h))
        return out

    # ------------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.crop_to_bbox and s.bbox is not None:
            x, y, w, h = s.bbox
            img = img.crop((x, y, x + w, y + h))     # crop to the bird before transforms
        img = self.transform(img)
        if self.return_bbox:
            return img, s.label, s.bbox
        return img, s.label
