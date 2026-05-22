import cv2
import numpy as np


def skin_mask_ycrcb(img_bgr: np.ndarray) -> np.ndarray:
    """Binary skin mask via YCrCb thresholding (Hsu et al. ranges)."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return cv2.medianBlur(mask, 5)


def skin_ratio_in_bbox(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """Fraction of skin pixels inside `bbox` (clipped to mask bounds)."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    return float(region.mean()) / 255.0
