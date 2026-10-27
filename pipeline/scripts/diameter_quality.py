"""Optional acquisition-ROI support check, independent of diameter geometry.

Call only for a reviewed acquisition viewport or an explicitly supplied ROI.
The check rejects short detected fragments; it cannot identify vessel anatomy.
It contains no temporal filter, diameter range, or physiological assumptions.
"""
from copy import deepcopy
import math


SPATIAL_SUPPORT_GATE_VERSION = 'minimum_longitudinal_roi_support_v1'
VERSION = SPATIAL_SUPPORT_GATE_VERSION


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_numbers(value):
    """Keep nonfinite diagnostic numbers out of strict JSON serialization."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _safe_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_numbers(item) for item in value]
    return value


def apply_spatial_support_gate(measurement, roi, min_axis_coverage=.35):
    """Return a copied, JSON-safe result with optional spatial support rejection.

    ROI is [left, top, right, bottom] in the same pixel coordinates as the
    measurement. The projected ROI span is abs(axis_x)*width+abs(axis_y)*height.
    Equality at the threshold passes. A previously unmeasurable result remains
    unmeasurable, even when sufficient support is present.

    Missing support metadata cannot establish eligibility for an otherwise
    successful result. Malformed configuration raises ValueError.
    """
    if not isinstance(measurement, dict):
        raise ValueError('measurement must be a dictionary')
    minimum = _finite(min_axis_coverage)
    if minimum is None or not 0 <= minimum <= 1:
        raise ValueError('min_axis_coverage must be finite and between 0 and 1')
    if roi is not None:
        try:
            bounds = [_finite(value) for value in roi]
        except TypeError as exc:
            raise ValueError('roi must contain four finite coordinates') from exc
        if (len(bounds) != 4 or any(value is None for value in bounds)
                or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
            raise ValueError('roi must contain ordered finite coordinates')
    else:
        bounds = None

    result = _safe_numbers(deepcopy(measurement))
    span = _finite(result.get('pca_longitudinal_extent_px'))
    axis = result.get('principal_axis')
    ax = ay = None
    if isinstance(axis, (list, tuple)) and len(axis) == 2:
        ax, ay = _finite(axis[0]), _finite(axis[1])
    projected = ratio = None
    reason = 'support_metadata_missing'
    if bounds is not None and span is not None and span >= 0 and ax is not None and ay is not None:
        norm = math.hypot(ax, ay)
        if norm > 0:
            projected = abs(ax/norm)*(bounds[2]-bounds[0])+abs(ay/norm)*(bounds[3]-bounds[1])
            ratio = span/projected
            reason = 'sufficient_longitudinal_support' if ratio >= minimum else 'insufficient_longitudinal_support'
    passed = ratio is not None and ratio >= minimum
    previous = result.get('spatial_support_gate', {})
    gate = {
        'version': SPATIAL_SUPPORT_GATE_VERSION,
        'axis_span_px': span,
        'projected_roi_span_px': projected,
        'axis_coverage': ratio,
        'min_axis_coverage': minimum,
        'passed': bool(passed),
        'evaluated': ratio is not None,
        'reason': reason,
        'roi': bounds,
    }
    # Retain candidate diagnostics on a repeated check without reviving a D.
    if isinstance(previous, dict):
        for name in ('candidate_mean_diameter_px', 'candidate_mean_diameter_mm',
                     'candidate_status', 'candidate_scanline_count'):
            if name in previous:
                gate[name] = previous[name]
    result['spatial_support_gate'] = gate
    result.setdefault('quality_gate_rejected_scanline_count', 0)
    if result.get('status') == 'ok' and not passed:
        lines = result.get('scanlines') or []
        gate.update(candidate_mean_diameter_px=result.get('mean_diameter_px'),
                    candidate_mean_diameter_mm=result.get('mean_diameter_mm'),
                    candidate_status=result.get('status'), candidate_scanline_count=len(lines))
        result['candidate_scanlines'] = lines
        result['scanlines'] = []
        result['scanline_count'] = 0
        result['coverage'] = 0.0
        result['quality_gate_rejected_scanline_count'] = len(lines)
        result['excluded_scanline_count'] = int(result.get('excluded_scanline_count') or 0)+len(lines)
        result['measured_range'] = None
        result['mean_diameter_px'] = None
        result['mean_diameter_mm'] = None
        result['status'] = 'unmeasurable'
        result['message'] = ('검출한 혈관 길이가 부족하여 측정하지 않습니다.' if ratio is not None
                             else '혈관 길이 평가에 필요한 정보가 부족하여 측정하지 않습니다.')
        warnings = list(result.get('warnings') or [])
        warnings.append(result['message'])
        result['warnings'] = warnings
    return result
