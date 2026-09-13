from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def retinaface_bbox(image_rgb: np.ndarray) -> np.ndarray | None:
    try:
        from insightface.app import FaceAnalysis

        if not hasattr(retinaface_bbox, "_app"):
            retinaface_bbox._app = FaceAnalysis(name="buffalo_l")
            retinaface_bbox._app.prepare(
                ctx_id=0 if torch.cuda.is_available() else -1,
                det_size=(640, 640),
            )
        faces = retinaface_bbox._app.get(image_rgb)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda item: (item.bbox[2] - item.bbox[0])
            * (item.bbox[3] - item.bbox[1]),
        )
        x0, y0, x1, y1 = [int(value) for value in face.bbox]
        height, width = image_rgb.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        return mask
    except Exception:
        return None


def robust_face_bbox_mask(
    image: Image.Image, expand: float = 0.15, min_conf: float = 0.30
) -> np.ndarray:
    image_array = np.array(image.convert("RGB"))
    height, width = image_array.shape[:2]

    mask = retinaface_bbox(image_array)
    if mask is not None:
        ys, xs = np.where(mask > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        expand_width = int((x1 - x0) * expand)
        expand_height = int((y1 - y0) * expand)
        x0, y0 = max(0, x0 - expand_width), max(0, y0 - expand_height)
        x1, y1 = min(width, x1 + expand_width), min(height, y1 + expand_height)
        output = np.zeros((height, width), dtype=np.uint8)
        output[y0:y1, x0:x1] = 1
        return output

    import mediapipe as mp

    with mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_conf
    ) as face_detector:
        result = face_detector.process(np.ascontiguousarray(image_array))
    if result and result.detections:
        detection = max(result.detections, key=lambda item: float(item.score[0]))
        box = detection.location_data.relative_bounding_box
        x0 = int((box.xmin - expand) * width)
        y0 = int((box.ymin - expand) * height)
        x1 = int((box.xmin + box.width + expand) * width)
        y1 = int((box.ymin + box.height + expand) * height)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 > x0 and y1 > y0:
            output = np.zeros((height, width), dtype=np.uint8)
            output[y0:y1, x0:x1] = 1
            return output

    side = int(0.7 * min(width, height))
    center_x, center_y = width // 2, height // 2
    x0, y0 = max(0, center_x - side // 2), max(0, center_y - side // 2)
    x1, y1 = min(width, x0 + side), min(height, y0 + side)
    output = np.zeros((height, width), dtype=np.uint8)
    output[y0:y1, x0:x1] = 1
    return output


class FaceMaskCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, image_path: str | Path) -> Path:
        return self.cache_dir / f"{Path(image_path).stem}.npy"

    def get(self, image_path: str | Path, image: Image.Image | None = None) -> np.ndarray:
        cache_path = self._cache_path(image_path)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                pass
        if image is None:
            image = Image.open(image_path).convert("RGB")
        mask = robust_face_bbox_mask(image)
        np.save(cache_path, mask.astype(np.uint8))
        return mask


def soften_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask

