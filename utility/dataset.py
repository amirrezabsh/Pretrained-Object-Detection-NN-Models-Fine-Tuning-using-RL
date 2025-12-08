import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from torchvision.datasets import VOCDetection
except ImportError:  # pragma: no cover - optional dependency
    VOCDetection = None


@dataclass
class DetectionSample:
    """
    Container for a single detection sample returned by `DetectionDataset`.

    Attributes
    ----------
    image : np.ndarray
        Raw RGB image as a HxWx3 float32 array in [0, 1].
    gt_boxes : np.ndarray
        Array with shape (N, 4) storing [x1, y1, x2, y2] pixel coordinates.
    metadata : dict
        Optional metadata (e.g., class labels) associated with the sample.
    """

    image: np.ndarray
    gt_boxes: np.ndarray
    metadata: Optional[dict] = None


def _read_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def _load_yolo_txt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Parse YOLO-format label file (one box per line: class cx cy w h in [0,1]).
    """
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)

    boxes: List[List[float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x_center = cx * img_w
            y_center = cy * img_h
            box_w = bw * img_w
            box_h = bh * img_h
            x1 = max(0.0, x_center - box_w / 2.0)
            y1 = max(0.0, y_center - box_h / 2.0)
            x2 = min(float(img_w), x_center + box_w / 2.0)
            y2 = min(float(img_h), y_center + box_h / 2.0)
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32)


class DetectionDataset:
    """
    Minimal dataset that yields (image, gt_boxes) tuples for the RL environment.

    The loader assumes YOLO-style annotations (TXT files with normalized coords),
    but it can also read a JSON file per image if `annotation_format="json"` is
    specified. JSON annotations must store a `boxes` list with [x1, y1, x2, y2].
    """

    def __init__(
        self,
        image_dir: Path | str,
        label_dir: Optional[Path | str] = None,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
        annotation_format: str = "yolo_txt",
        limit: Optional[int] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else self.image_dir
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.annotation_format = annotation_format

        self.image_paths = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in self.extensions
            ]
        )
        if limit:
            self.image_paths = self.image_paths[:limit]
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.image_paths[idx]
        image = _read_image(img_path)
        h, w = image.shape[:2]

        if self.annotation_format == "yolo_txt":
            label_path = (self.label_dir / img_path.stem).with_suffix(".txt")
            gt_boxes = _load_yolo_txt_boxes(label_path, w, h)
            metadata = None
        elif self.annotation_format == "json":
            label_path = (self.label_dir / img_path.stem).with_suffix(".json")
            metadata = json.loads(label_path.read_text(encoding="utf-8"))
            gt_boxes = np.asarray(metadata.get("boxes", []), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported annotation_format: {self.annotation_format}")

        return image, gt_boxes


class VOCDetectionDataset:
    """
    Wrap torchvision's VOCDetection to emit numpy images and xyxy boxes.
    """

    def __init__(
        self,
        root: Path | str,
        image_set: str = "trainval",
        year: str = "2007",
        download: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        if VOCDetection is None:
            raise ImportError(
                "torchvision is required for VOC datasets. Install torchvision to continue."
            )
        self.dataset = VOCDetection(
            root=str(root),
            year=year,
            image_set=image_set,
            download=download,
        )
        self.limit = limit

    def __len__(self) -> int:
        length = len(self.dataset)
        if self.limit is not None:
            return min(length, self.limit)
        return length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.limit is not None:
            idx = min(idx, self.limit - 1)
        image_pil, target = self.dataset[idx]
        image = np.asarray(image_pil).astype(np.float32) / 255.0

        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        boxes: List[List[float]] = []
        for obj in objects:
            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"])
            y1 = float(bbox["ymin"])
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])
            boxes.append([x1, y1, x2, y2])

        gt_boxes = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
        return image, gt_boxes


def _voc_dataset_exists(root: Path, image_set: Optional[str] = None) -> bool:
    """
    Check if Pascal VOC 2007 files are present under the expected directory.
    """

    return _locate_voc_root(root, image_set=image_set) is not None


def _locate_voc_root(root: Path, image_set: Optional[str] = None) -> Optional[Path]:
    """
    Return the directory that directly contains VOCdevkit (if found), supporting
    common archive extraction layouts.
    """

    root = Path(root)
    # Try common layouts: either directly under root/VOCdevkit or nested one level
    # deeper (e.g., root/VOCtrainval_06-Nov-2007/VOCdevkit).
    candidates = [root] + sorted([p for p in root.glob("*") if p.is_dir()])

    def _candidate_valid(candidate: Path, require_split: bool) -> bool:
        voc_root = candidate / "VOCdevkit" / "VOC2007"
        annotations = voc_root / "Annotations"
        images = voc_root / "JPEGImages"
        if not (annotations.exists() and images.exists()):
            return False
        if require_split and image_set:
            split_file = voc_root / "ImageSets" / "Main" / f"{image_set}.txt"
            return split_file.exists()
        return True

    # Prefer candidates that include the requested split file to avoid picking the
    # wrong archive (e.g., VOCtest when requesting trainval).
    for candidate in candidates:
        if _candidate_valid(candidate, require_split=True):
            return candidate

    # Fallback to any candidate with annotations/images if the specific split is missing.
    for candidate in candidates:
        if _candidate_valid(candidate, require_split=False):
            return candidate
    return None


def load_custom_dataset(
    image_dir: Path | str,
    label_dir: Optional[Path | str] = None,
    limit: Optional[int] = None,
    annotation_format: str = "yolo_txt",
) -> DetectionDataset:
    """
    Convenience helper to keep notebooks minimal.

    Example
    -------
    >>> dataset = load_custom_dataset(\"data/images\", \"data/labels\")
    >>> img, gt = dataset[0]
    """

    return DetectionDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        annotation_format=annotation_format,
        limit=limit,
    )


def load_pascal_voc2007(
    root: Path | str,
    image_set: str = "trainval",
    limit: Optional[int] = 500,
    download: bool = False,
) -> VOCDetectionDataset:
    """
    Convenience helper for a lightweight Pascal VOC 2007 subset.

    Parameters
    ----------
    root : str | Path
        Directory where VOC2007 will be stored (same as torchvision's root).
    image_set : str
        One of \"train\", \"val\", \"trainval\", or \"test\".
    limit : Optional[int]
        Cap the number of samples returned to keep experiments lightweight.
    download : bool
        Attempt to download VOC if it's missing. When True, a download is only
        triggered if the dataset is not already present.
    """

    root_path = Path(root)
    resolved_root = _locate_voc_root(root_path, image_set=image_set)
    dataset_exists = resolved_root is not None
    should_download = download and not dataset_exists
    target_root = resolved_root if resolved_root is not None else root_path

    if not dataset_exists and not should_download:
        raise FileNotFoundError(
            f"Pascal VOC 2007 not found under {root_path}. Set download=True to fetch it automatically."
        )

    return VOCDetectionDataset(
        root=target_root,
        image_set=image_set,
        year="2007",
        download=should_download,
        limit=limit,
    )
