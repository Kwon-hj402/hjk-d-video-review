"""Automatic original-pixel SAM boxes for provisional ultrasound mask drafts.

This module finds a smooth dark-band path in the scan area. A dark-band path is
an image-processing prompt proxy, not an anatomical finding or ground truth.
Every valid image receives a box, including poorly visible frames. Diagnostics
describe the prompt heuristic only; successful SAM execution is not accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np

PROMPT_VERSION = "dark_band_viterbi_v1"


def _close_small_gaps(mask: np.ndarray, max_gap: int = 7) -> np.ndarray:
    """Bridge small speckle interruptions along a one-dimensional column."""
    result = mask.copy()
    starts = np.flatnonzero(np.diff(np.r_[False, ~mask, False].astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, ~mask, False].astype(np.int8)) == -1)
    for start, end in zip(starts, ends):
        if 0 < start and end < len(mask) and end - start <= max_gap:
            result[start:end] = True
    return result


def _column_candidates(profile: np.ndarray, threshold: float, y0: int, scale_y: float):
    mask = _close_small_gaps(profile < threshold, max_gap=max(2, round(7 * scale_y)))
    starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)) == -1)
    candidates = []
    margin = max(3, round(8 * scale_y))
    for start, end in zip(starts, ends):
        thickness = end - start
        if thickness < 18 * scale_y or thickness > 168 * scale_y:
            continue
        center = y0 + (start + end) / 2
        if center < 165 * scale_y or center > 342 * scale_y:
            continue
        mean = float(np.mean(profile[start:end]))
        top = float(np.mean(profile[max(0, start-margin):start])) if start > 0 else mean
        bottom = float(np.mean(profile[end:min(len(profile), end+margin)])) if end < len(profile) else mean
        edge_contrast = min(top, bottom) - mean
        boundary_visible = start > 2 * scale_y and end < len(profile) - 2 * scale_y
        score = (
            .55 * min(thickness / (62 * scale_y), 1)
            + .65 * np.clip(edge_contrast / 45, 0, 1)
            + .35 * np.clip(1 - mean / max(threshold, 1), 0, 1)
            - .17 * max(0, (thickness - 112 * scale_y) / (50 * scale_y))
            - .16 * abs(center / scale_y - 260) / 110
            - (0 if boundary_visible else .35)
        )
        candidates.append({
            "top": float(y0 + start), "bottom": float(y0 + end), "center": center,
            "width": float(thickness), "score": float(score),
            "mean": mean, "edge_contrast": edge_contrast,
            "real_dark_run": True, "boundary_visible": bool(boundary_visible),
        })
    # Sparse/obscured columns still need a prompt for the all-frame SAM run.
    # Add a deliberately lower-scoring fallback dark window. It carries no
    # assertion that a vessel boundary is visible at its edges.
    window = max(12, round(74 * scale_y))
    cumulative = np.r_[0.0, np.cumsum(profile, dtype=float)]
    means = (cumulative[window:] - cumulative[:-window]) / window
    centers = y0 + np.arange(len(means)) + window / 2
    objective = means + 5 * ((centers / scale_y - 265) / 100) ** 2
    valid = (centers >= 185 * scale_y) & (centers <= 330 * scale_y)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        start = int(valid_indices[np.argmin(objective[valid_indices])])
        candidates.append({
            "top": float(y0 + start), "bottom": float(y0 + start + window),
            "center": float(y0 + start + window / 2), "width": float(window),
            "score": -.25 + .15 * float(np.clip(1 - means[start] / 40, 0, 1)),
            "mean": float(means[start]), "edge_contrast": 0.0,
            "real_dark_run": False, "boundary_visible": False,
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:7]


def make_prompt(frame: dict | None, image: np.ndarray) -> tuple[list[int], dict]:
    """Return ``([x1,y1,x2,y2], diagnostics)`` in the uncropped image coordinates.

    ``image`` accepts cv2 BGR/BGRA or grayscale arrays. ``frame`` is the manifest
    entry, used only for a traceable frame identifier. No source-specific anatomy
    or timing label is inferred. The full frame must be supplied to SAM.
    """
    if image is None or image.ndim not in (2, 3) or min(image.shape[:2]) < 32:
        raise ValueError("A nonempty original image is required for an automatic prompt")
    height, width = image.shape[:2]
    if image.ndim == 2:
        gray = image.astype(np.uint8, copy=False)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx, sy = width / 640, height / 448
    x0, x1 = round(120 * sx), round(525 * sx)
    y0, y1 = round(150 * sy), round(360 * sy)
    kernel = max(3, round(9 * min(sx, sy)))
    kernel += 1 - kernel % 2
    smooth = cv2.GaussianBlur(gray, (kernel, kernel), 2.0 * min(sx, sy))
    scan = smooth[y0:y1, x0:x1]
    threshold = float(np.clip(np.percentile(scan, 35) * .55, 8, 28))
    xs = np.linspace(round(130 * sx), round(510 * sx), 29).round().astype(int)
    half_width = max(2, round(5 * sx))
    columns = []
    for x in xs:
        profile = smooth[y0:y1, max(0, x-half_width):min(width, x+half_width+1)].mean(axis=1)
        columns.append(_column_candidates(profile, threshold, y0, sy))
    # Smoothness prevents jumping between isolated dark tissue and separate
    # branches. Viterbi chooses the strongest smooth dark-band path across x.
    costs = [np.asarray([c["score"] for c in columns[0]], dtype=float)]
    parents = [np.full(len(columns[0]), -1, dtype=int)]
    for col_index in range(1, len(columns)):
        prior, current = columns[col_index-1], columns[col_index]
        values, back = [], []
        for candidate in current:
            transitions = np.asarray([
                costs[-1][j]
                - .009 * abs(candidate["center"] - old["center"]) / sy
                - .004 * abs(candidate["width"] - old["width"]) / sy
                for j, old in enumerate(prior)
            ])
            winner = int(np.argmax(transitions))
            values.append(float(transitions[winner] + candidate["score"]))
            back.append(winner)
        costs.append(np.asarray(values))
        parents.append(np.asarray(back, dtype=int))
    chosen = int(np.argmax(costs[-1]))
    path = []
    for col_index in reversed(range(len(columns))):
        path.append(columns[col_index][chosen])
        chosen = int(parents[col_index][chosen])
    path.reverse()
    real = [candidate for candidate in path if candidate["real_dark_run"]]
    reliable_edges = [candidate for candidate in real if candidate["boundary_visible"] and candidate["edge_contrast"] >= 5]
    bounds = reliable_edges if len(reliable_edges) >= 10 else real if len(real) >= 8 else path
    tops = np.asarray([candidate["top"] for candidate in bounds])
    bottoms = np.asarray([candidate["bottom"] for candidate in bounds])
    # Keep near/far walls inside the SAM prompt and exclude ECG/side panels.
    top = int(np.floor(np.percentile(tops, 8) - 14 * sy))
    bottom = int(np.ceil(np.percentile(bottoms, 92) + 18 * sy))
    top = max(round(140 * sy), min(top, round(315 * sy)))
    bottom = min(round(378 * sy), max(bottom, top + round(38 * sy)))
    box = [max(0, round(110 * sx)), top, min(width-1, round(535 * sx)), bottom]
    diagnostics = {
        "prompt_method": PROMPT_VERSION,
        "frame_id": frame.get("id") if frame else None,
        "coordinate_system": "original_uncropped_image_pixels",
        "automatic_box": box,
        "search_roi": [x0, y0, x1, y1],
        "dark_threshold": round(threshold, 3),
        "dark_run_column_fraction": round(len(real) / len(path), 4),
        "visible_edge_column_fraction": round(len(reliable_edges) / len(path), 4),
        "median_band_thickness_px": round(float(np.median([candidate["width"] for candidate in path])), 3),
        "median_band_center_y_px": round(float(np.median([candidate["center"] for candidate in path])), 3),
        "fallback_used": len(real) < 8,
        "requires_human_review": True,
        "is_anatomical_or_accuracy_validation": False,
    }
    return box, diagnostics


def automatic_box(frame: dict | None, image: np.ndarray) -> list[int]:
    """Return only the original-coordinate SAM box; preferred batch signature."""
    return make_prompt(frame, image)[0]
