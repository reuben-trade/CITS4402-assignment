import cv2
import numpy as np


def apply_clahe_ycrcb(
    img_bgr: np.ndarray,
    clip_limit: float = 3.0,
    tile: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Lift shadows on dark images by running CLAHE on the Y channel of YCrCb.

    Colour information (Cr, Cb) is preserved so skin tones stay realistic.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    y_eq = clahe.apply(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
