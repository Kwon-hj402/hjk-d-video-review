"""Conservative left-edge depth-tick calibration for the current scan layout.

No fixed pixel pitch is substituted when markers cannot be verified. The caller
must know the physical separation of adjacent displayed ticks (tick_mm).
"""
import math

import cv2
import numpy as np


def detect_tick_calibration(image, tick_mm=5.0):
    """Return JSON-safe scale metadata from original-resolution BGR/gray pixels.

    The current device puts short AND long adjacent depth ticks against the left
    edge. Thus long-line spacing must not be mistaken for adjacent-line spacing.
    Different layouts or missing/uneven ticks produce available=False, requiring
    pixel-only measurements or an explicit user-provided scale.
    """
    result = {"available": False, "pitch_px": None, "mm_per_pixel": None,
              "tick_centers": [], "reason": "", "method": "left_adjacent_depth_ticks_v1"}
    try:
        spacing_mm = float(tick_mm)
    except (TypeError, ValueError):
        result["reason"] = "tick_mm must be finite and greater than zero"
        return result
    if not math.isfinite(spacing_mm) or spacing_mm <= 0:
        result["reason"] = "tick_mm must be finite and greater than zero"
        return result
    result["tick_mm"] = spacing_mm
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        result["reason"] = "An original uint8 grayscale or BGR image is required"
        return result
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        result["reason"] = "Unsupported image shape"
        return result
    height, width = gray.shape
    if height < 80 or width < 32:
        result["reason"] = "Image is too small to verify depth markers"
        return result
    # Scaled from the reviewed 640x448 layout: x=0..6, y=30..390.
    strip_width = max(3, int(round(width * 7 / 640)))
    y_start = max(0, int(round(height * 30 / 448)))
    y_end = min(height, int(round(height * 391 / 448)))
    strip = gray[y_start:y_end, :strip_width]
    result["roi"] = [0, y_start, strip_width, y_end]
    peak = float(strip.max())
    background = float(np.median(strip))
    if peak - background < 35:
        result["reason"] = "Insufficient contrast in the depth-marker strip"
        return result
    threshold = max(40.0, background + 0.45 * (peak - background))
    bright = strip >= threshold
    # A tick must reach the frame edge and extend horizontally; isolated speckle
    # and a narrow vertical border do not count as scale markers.
    minimum_width = max(2, int(math.ceil(strip_width * 0.35)))
    rows = bright[:, 0] & (bright.sum(axis=1) >= minimum_width)
    padded = np.r_[False, rows, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    max_thickness = max(3, int(round(height * 4 / 448)))
    centers = [float(y_start + (a + b - 1) / 2) for a, b in zip(starts, ends)
               if b - a <= max_thickness]
    result["tick_centers"] = centers
    if len(centers) < 4:
        result["reason"] = "Fewer than four isolated adjacent depth ticks found"
        return result
    gaps = np.diff(centers)
    median_gap = float(np.median(gaps))
    if median_gap < 8 or float(gaps.min()) < 8:
        result["reason"] = "Depth-marker pitch is less than eight pixels"
        return result
    relative_gap_error = float(np.max(np.abs(gaps - median_gap)) / median_gap)
    if relative_gap_error > 0.18:
        result["reason"] = "Depth ticks have missing, additional, or uneven spacing"
        result["max_relative_gap_variation"] = relative_gap_error
        return result
    indices = np.arange(len(centers), dtype=np.float64)
    fit_pitch, intercept = np.polyfit(indices, np.asarray(centers), 1)
    residual = np.asarray(centers) - (intercept + fit_pitch * indices)
    # Average all verified adjacent intervals, rather than rounding alternating
    # one/two-row marker centers or measuring long ticks alone.
    pitch = float(np.mean(gaps))
    max_residual = float(np.max(np.abs(residual)))
    if not np.isfinite(pitch) or pitch < 8 or max_residual > max(1.5, float(pitch) * 0.08):
        result["reason"] = "Depth ticks do not fit an evenly spaced scale"
        return result
    result.update(available=True, pitch_px=float(pitch), mm_per_pixel=float(spacing_mm / pitch),
                  max_relative_gap_variation=relative_gap_error, max_fit_residual_px=max_residual,
                  reason="At least four isolated, evenly spaced adjacent ticks verified")
    return result
