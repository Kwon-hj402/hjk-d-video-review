"""Spatial diameter perpendicular to a smooth local centerline, frame by frame."""
import math
import statistics

import cv2
import numpy as np

METHOD = 'local_centerline_normal_wall_to_wall_middle_70_percent_v1'


def _result(status='unmeasurable', message='측정할 내강 경계가 없습니다.'):
    return {
        'status': status, 'message': message, 'method': METHOD,
        'mean_diameter_px': None, 'mean_diameter_mm': None, 'sd_diameter_px': None,
        'scanline_count': 0, 'sample_count': 0, 'excluded_scanline_count': 0,
        'outlier_scanline_count': 0, 'rapid_change_excluded_scanline_count': 0,
        'branch_excluded_scanline_count': 0, 'coverage': 0.0,
        'scanlines': [], 'rejected_scanlines': [], 'centerline': [],
        'centerline_origin': None, 'principal_axis': None, 'angle_degrees': None,
        'measured_range': None, 'requested_range': None, 'roi': None,
        'warnings': [], 'filled_hole_count': 0, 'filled_hole_pixels': 0,
        'ignored_component_count': 0, 'component_area_px': 0,
    }


def _smooth(values, sigma):
    radius = max(2, int(math.ceil(3 * sigma)))
    positions = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-positions ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, (radius, radius), mode='edge'), kernel, mode='valid')


def _intervals(samples, coordinates, threshold=.5):
    """Continuous .5-level crossings of an interpolated binary mask ray."""
    transitions = np.diff(np.r_[False, samples >= threshold, False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)-1
    intervals = []
    for start, end in zip(starts, ends):
        if start == 0 or end == len(samples)-1:
            continue
        a0, a1 = float(samples[start-1]), float(samples[start])
        b0, b1 = float(samples[end]), float(samples[end+1])
        left = float(coordinates[start-1] + (threshold-a0)/max(a1-a0, 1e-12)*(coordinates[start]-coordinates[start-1]))
        right = float(coordinates[end] + (b0-threshold)/max(b0-b1, 1e-12)*(coordinates[end+1]-coordinates[end]))
        if right-left >= 1.5:
            intervals.append((left, right))
    return intervals


def _small_holes(component):
    cleaned = component.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~component).astype(np.uint8), 8)
    total, pixels = 0, 0
    height, width = component.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x+w == width or y+h == height
        if not touches_border and area <= 25 and max(w, h) <= 8:
            cleaned[labels == label] = True
            total += 1
            pixels += int(area)
    return cleaned, total, pixels


def _robust_limit(values, relative=.15, minimum=2.0):
    median = float(np.median(values))
    mad = float(np.median(np.abs(values-median)))
    return median, max(minimum, relative*median, 3.5*1.4826*mad)


def measure_mask(mask, mm_per_pixel=None, roi=None):
    """Local-normal wall-to-wall widths from the central 70% of a main mask.

    For 640×448 frames the default scan ROI is [94,40,545,382]; other sizes use
    the whole image. Only a copy can have small internal holes filled. PCA is
    used for alignment, never as the diameter estimator. No time smoothing,
    temporal rejection, FMD computation, or input-mask mutation takes place.
    The returned dict is JSON-safe. Missing/unstable geometry has D=None.
    """
    result = _result()
    try:
        array = np.asarray(mask)
        if array.ndim == 3 and 1 in array.shape:
            array = np.squeeze(array)
        if array.ndim != 2 or min(array.shape) < 8 or not np.all(np.isfinite(array)):
            return _result('invalid', '유효한 2차원 이진 마스크가 필요합니다.')
        height, width = array.shape
        binary = ((array >= .5) if np.issubdtype(array.dtype, np.floating) else (array > 0)).copy()
        bounds = [94, 40, 545, 382] if roi is None and (height, width) == (448, 640) else ([0, 0, width, height] if roi is None else list(roi))
        if len(bounds) != 4 or not all(math.isfinite(float(value)) for value in bounds):
            return _result('invalid', 'ROI 좌표가 유효하지 않습니다.')
        x0, y0 = max(0, int(math.floor(bounds[0]))), max(0, int(math.floor(bounds[1])))
        x1, y1 = min(width, int(math.ceil(bounds[2]))), min(height, int(math.ceil(bounds[3])))
        if x1-x0 < 8 or y1-y0 < 8:
            return _result('invalid', '측정 ROI가 너무 작습니다.')
        result['roi'] = [x0, y0, x1, y1]
        restricted = np.zeros_like(binary)
        restricted[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(restricted.astype(np.uint8), 8)
        if count <= 1:
            result['message'] = 'ROI 안에서 내강 마스크를 찾지 못했습니다.'
            return result
        main = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = labels == main
        area = int(stats[main, cv2.CC_STAT_AREA])
        result['component_area_px'], result['ignored_component_count'] = area, count-2
        if area < 60:
            result['message'] = '내강 마스크가 너무 작아 직경을 측정할 수 없습니다.'
            return result
        component, filled_count, filled_pixels = _small_holes(component)
        result['filled_hole_count'], result['filled_hole_pixels'] = filled_count, filled_pixels
        ys, xs = np.nonzero(component)
        points = np.column_stack((xs, ys)).astype(np.float64)
        origin = points.mean(axis=0)
        centered = points-origin
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T@centered/len(points))
        axis = eigenvectors[:, -1]
        if axis[0] < 0 or (abs(axis[0]) < 1e-10 and axis[1] < 0):
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        t_points, n_points = centered@axis, centered@normal
        lo, hi = float(t_points.min()), float(t_points.max())
        span = hi-lo
        result.update(centerline_origin=origin.tolist(), principal_axis=axis.tolist(),
                      angle_degrees=float(math.degrees(math.atan2(axis[1], axis[0]))),
                      pca_longitudinal_extent_px=span,
                      pca_eigenvalue_ratio=float(eigenvalues[-1]/max(eigenvalues[0], 1e-12)))
        if span < 30 or eigenvalues[0] <= .2 or result['pca_eigenvalue_ratio'] < 2.0:
            result['message'] = '길쭉한 내강 구간과 두 벽을 구분할 수 없습니다.'
            return result
        start, end = lo+.15*span, hi-.15*span
        result['requested_range'] = dict(start_t=start, end_t=end, end_trim_fraction_each=.15,
                                        start_point=(origin+axis*start).tolist(), end_point=(origin+axis*end).tolist())
        ts = np.arange(math.floor(lo)-2, math.ceil(hi)+3, 1.0)
        ns = np.arange(math.floor(float(n_points.min()))-3, math.ceil(float(n_points.max()))+4, 1.0)
        map_x = (origin[0]+axis[0]*ts[None, :]+normal[0]*ns[:, None]).astype(np.float32)
        map_y = (origin[1]+axis[1]*ts[None, :]+normal[1]*ns[:, None]).astype(np.float32)
        interpolation_mask = component.astype(np.float32)
        aligned = cv2.remap(interpolation_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tops, bottoms = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
        reasons = np.full(len(ts), 'no_wall_pair', dtype=object)
        for index in range(len(ts)):
            crossings = _intervals(aligned[:, index], ns)
            if len(crossings) > 1:
                reasons[index] = 'multiple_intersections'
            elif len(crossings) == 1:
                top, bottom = crossings[0]
                if bottom-top >= 3:
                    tops[index], bottoms[index], reasons[index] = top, bottom, ''
        central, valid = (ts >= start)&(ts <= end), np.isfinite(tops)&np.isfinite(bottoms)
        widths = bottoms-tops
        if np.count_nonzero(valid&central) < 12:
            result['message'] = '중앙 구간에서 단일한 두 벽의 교차를 충분히 찾지 못했습니다.'
            return result
        axis_median, axis_limit = _robust_limit(widths[valid&central], relative=.15, minimum=3)
        if axis_median < 4:
            result['message'] = '마스크 두께가 너무 얇아 안정적으로 측정할 수 없습니다.'
            return result
        raw_outliers = valid&(np.abs(widths-axis_median) > axis_limit)
        top_interp = np.interp(ts, ts[valid], tops[valid])
        bottom_interp = np.interp(ts, ts[valid], bottoms[valid])
        width_gradient = np.abs(np.gradient(_smooth(bottom_interp-top_interp, 3)))
        top_curvature = np.abs(np.gradient(np.gradient(_smooth(top_interp, 3))))
        bottom_curvature = np.abs(np.gradient(np.gradient(_smooth(bottom_interp, 3))))
        stable = central&valid&~raw_outliers
        median_gradient = float(np.median(width_gradient[stable])) if np.any(stable) else 0.0
        gradient_mad = float(np.median(np.abs(width_gradient[stable]-median_gradient))) if np.any(stable) else 0.0
        gradient_limit = max(.7, median_gradient+8*1.4826*gradient_mad)
        curvature = np.maximum(top_curvature, bottom_curvature)
        median_curvature = float(np.median(curvature[stable])) if np.any(stable) else 0.0
        curvature_mad = float(np.median(np.abs(curvature[stable]-median_curvature))) if np.any(stable) else 0.0
        curvature_limit = max(.18, median_curvature+8*1.4826*curvature_mad)
        rapid = (width_gradient > gradient_limit)|(curvature > curvature_limit)
        guard_radius = max(2, min(5, round(axis_median*.05)))
        rapid = np.convolve(rapid.astype(np.uint8), np.ones(2*guard_radius+1, dtype=np.uint8), mode='same') > 0
        fit_valid = valid&~raw_outliers&~rapid
        if np.count_nonzero(fit_valid&central) < 12:
            result['message'] = '분지나 급변하는 경계를 제외한 중심선 구간이 부족합니다.'
            return result
        centers = (tops+bottoms)/2
        center_values = np.interp(ts, ts[fit_valid], centers[fit_valid])
        center_values = _smooth(center_values, max(2, min(8, span*.015)))
        slopes = np.gradient(center_values)
        sample_ts = np.linspace(start, end, min(81, max(12, int((end-start)/2)+1)))
        result['sample_count'] = len(sample_ts)
        result['centerline'] = [(origin+axis*t+normal*float(np.interp(t, ts, center_values))).tolist() for t in sample_ts]
        provisional = []
        ray_extent = math.hypot(float(xs.max()-xs.min()), float(ys.max()-ys.min()))+4
        ray_coordinates = np.arange(-ray_extent, ray_extent+.5, .5, dtype=np.float64)

        def reject(t, center, reason, **extra):
            entry = dict(t=float(t), x=float(center[0]), y=float(center[1]), reason=reason)
            entry.update(extra)
            result['rejected_scanlines'].append(entry)

        for t in sample_ts:
            index = int(np.argmin(np.abs(ts-t)))
            center = origin+axis*t+normal*float(np.interp(t, ts, center_values))
            if not valid[index]:
                reject(t, center, str(reasons[index])); continue
            if raw_outliers[index]:
                reject(t, center, 'spatial_width_outlier', axis_width_px=float(widths[index])); continue
            if rapid[index]:
                reject(t, center, 'rapid_boundary_change'); continue
            tangent = axis+normal*float(np.interp(t, ts, slopes))
            tangent /= np.linalg.norm(tangent)
            local_normal = np.array([-tangent[1], tangent[0]])
            ray_x = (center[0]+local_normal[0]*ray_coordinates).astype(np.float32)[None, :]
            ray_y = (center[1]+local_normal[1]*ray_coordinates).astype(np.float32)[None, :]
            samples = cv2.remap(interpolation_mask, ray_x, ray_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
            crossings = _intervals(samples, ray_coordinates)
            if len(crossings) != 1:
                reject(t, center, 'multiple_intersections' if len(crossings) > 1 else 'no_wall_pair'); continue
            left, right = crossings[0]
            if not left <= 0 <= right:
                reject(t, center, 'center_outside_lumen'); continue
            a, b = center+local_normal*left, center+local_normal*right
            diameter = float(right-left)
            if diameter < 3:
                reject(t, center, 'thin_segment'); continue
            if any(p[0] <= x0+.6 or p[0] >= x1-1.6 or p[1] <= y0+.6 or p[1] >= y1-1.6 for p in (a, b)):
                reject(t, center, 'roi_clipped_wall'); continue
            wall_ts = np.asarray([(a-origin)@axis, (b-origin)@axis])
            cap_guard = max(3, .025*span)
            if wall_ts.min() < lo+cap_guard or wall_ts.max() > hi-cap_guard:
                reject(t, center, 'end_cap_intersection'); continue
            provisional.append(dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]),
                                    x=float(center[0]), center_x=float(center[0]), center_y=float(center[1]),
                                    t=float(t), diameter_px=diameter, tangent_x=float(tangent[0]), tangent_y=float(tangent[1])))
        if provisional:
            ds = np.asarray([line['diameter_px'] for line in provisional])
            median, limit = _robust_limit(ds, relative=.12, minimum=2)
            for line in provisional:
                if abs(line['diameter_px']-median) > limit:
                    reject(line['t'], np.array([line['center_x'], line['center_y']]), 'normal_width_mad_outlier', diameter_px=line['diameter_px'])
                else:
                    result['scanlines'].append(line)
        result['scanline_count'] = len(result['scanlines'])
        result['excluded_scanline_count'] = len(result['rejected_scanlines'])
        result['outlier_scanline_count'] = sum(line['reason'] in ('spatial_width_outlier', 'normal_width_mad_outlier') for line in result['rejected_scanlines'])
        result['rapid_change_excluded_scanline_count'] = sum(line['reason'] == 'rapid_boundary_change' for line in result['rejected_scanlines'])
        result['branch_excluded_scanline_count'] = sum(line['reason'] == 'multiple_intersections' for line in result['rejected_scanlines'])
        result['coverage'] = result['scanline_count']/max(1, result['sample_count'])
        if result['scanline_count']:
            accepted_ts = [line['t'] for line in result['scanlines']]
            a_t, b_t = min(accepted_ts), max(accepted_ts)
            point_at = lambda t: (origin+axis*t+normal*float(np.interp(t, ts, center_values))).tolist()
            result['measured_range'] = dict(start_t=a_t, end_t=b_t, start_point=point_at(a_t), end_point=point_at(b_t), longitudinal_length_px=b_t-a_t)
        if result['ignored_component_count']:
            result['warnings'].append('ROI 안의 가장 큰 연결 마스크만 사용하고 별도 작은 마스크를 제외했습니다.')
        if filled_count:
            result['warnings'].append(f'외벽 폭을 위해 작은 내부 빈틈 {filled_count}개({filled_pixels}px)를 측정용 복사본에서만 메웠습니다.')
        if result['excluded_scanline_count']:
            result['warnings'].append('분지·다중교차·급변 경계·공간 폭 이상치를 제외했습니다.')
        if result['scanline_count'] < 12 or result['coverage'] < .45:
            result['message'] = '안정적인 두 벽 측정 구간이 부족해 직경을 표시하지 않습니다.'
            return result
        ds = [line['diameter_px'] for line in result['scanlines']]
        mean, sd = statistics.mean(ds), statistics.pstdev(ds)
        if sd/mean > .25:
            result['message'] = '남은 구간의 폭 차이가 커서 직경을 표시하지 않습니다.'
            return result
        result.update(status='ok', mean_diameter_px=mean, sd_diameter_px=sd)
        if isinstance(mm_per_pixel, (int, float, np.number)) and math.isfinite(float(mm_per_pixel)) and mm_per_pixel > 0:
            result['mean_diameter_mm'] = mean*float(mm_per_pixel)
        else:
            result['warnings'].append('mm/픽셀 길이 보정이 없어 픽셀 단위로만 계산합니다.')
        result['message'] = f"중앙 70%에서 중심선에 직각인 두 벽 사이 {result['scanline_count']}개 폭의 평균입니다."
        return result
    except (ValueError, TypeError, IndexError, cv2.error, np.linalg.LinAlgError) as error:
        result.update(status='invalid', message='마스크 기하를 계산할 수 없습니다.')
        result['warnings'].append(type(error).__name__)
        return result


def measure_annotation(annotation, frame):
    """Union all lumen polygons using even/odd raster fill, then mask geometry.

    Sampling polygons at x+.5,y+.5 avoids the inclusive-edge +1px bias for
    integer-coordinate parallel-wall polygons. Self intersections and multiple
    objects become a mask; separate components are handled by measure_mask.
    """
    if not isinstance(annotation, dict) or annotation.get('status') in ('excluded', 'negative') or annotation.get('is_negative') or annotation.get('negative'):
        return _result(message='음성·제외·미지정 레이블에는 직경을 계산하지 않습니다.')
    height, width = int(frame.get('height', 448)), int(frame.get('width', 640))
    mask = np.zeros((height, width), dtype=np.uint8)
    usable_objects = 0
    for obj in annotation.get('objects', []):
        if obj.get('class_id', 0) not in (0, '0'):
            continue
        try:
            points = np.asarray(obj.get('points', []), dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
                continue
            x_start = max(0, int(math.ceil(float(points[:, 0].min())-.5)))
            x_end = min(width-1, int(math.floor(float(points[:, 0].max())-.5)))
            y_start = max(0, int(math.ceil(float(points[:, 1].min())-.5)))
            y_end = min(height-1, int(math.floor(float(points[:, 1].max())-.5)))
            if x_end < x_start or y_end < y_start:
                continue
            grid_y, grid_x = np.mgrid[y_start:y_end+1, x_start:x_end+1]
            grid_x, grid_y = grid_x+.5, grid_y+.5
            inside = np.zeros(grid_x.shape, dtype=bool)
            for a, b in zip(points, np.roll(points, -1, axis=0)):
                if abs(b[1]-a[1]) < 1e-12:
                    continue
                crosses = (a[1] > grid_y) != (b[1] > grid_y)
                crossing_x = a[0]+(grid_y-a[1])*(b[0]-a[0])/(b[1]-a[1])
                inside ^= crosses & (grid_x < crossing_x)
            mask[y_start:y_end+1, x_start:x_end+1] |= inside.astype(np.uint8)
            usable_objects += 1
        except (TypeError, ValueError, IndexError):
            continue
    if not usable_objects or not np.any(mask):
        return _result(message='유효한 내강 경계 레이블이 없습니다.')
    result = measure_mask(mask, annotation.get('mm_per_pixel'))
    # Return ray/centerline positions in the original polygon coordinates.
    for line in result['scanlines']:
        for field in ('x1', 'y1', 'x2', 'y2', 'x', 'center_x', 'center_y'):
            line[field] += .5
    for line in result['rejected_scanlines']:
        line['x'] += .5
        line['y'] += .5
    result['centerline'] = [[x+.5, y+.5] for x, y in result['centerline']]
    if result['centerline_origin'] is not None:
        result['centerline_origin'] = [value+.5 for value in result['centerline_origin']]
    for range_field in ('measured_range', 'requested_range'):
        if result[range_field] is not None:
            for field in ('start_point', 'end_point'):
                result[range_field][field] = [value+.5 for value in result[range_field][field]]
    result['annotation_object_count'] = usable_objects
    return result


def summarize_fmd(rows):
    """Require calibrated, approved brachial frames and protocol timing."""
    sessions = {}
    for frame, ann in rows:
        key = (ann.get('patient_id', 'unassigned'), ann.get('session_id', 'unassigned'))
        if ann.get('status') != 'approved':
            continue
        sessions.setdefault(key, []).append((frame, ann))
    results = []
    for (patient, session), items in sessions.items():
        out = {'patient_id': patient, 'session_id': session, 'fmd_percent': None,
               'baseline_mm': None, 'peak_mm': None, 'issues': [], 'baseline_frames': [], 'peak_frames': []}
        if patient in ('', 'unassigned') or session in ('', 'unassigned'):
            out['issues'].append('대상자와 검사 세션을 지정하세요.')
        for frame, ann in items:
            if ann.get('anatomy') != 'brachial':
                out['issues'].append('상완동맥 촬영 확인이 필요합니다. 경동맥 자료에는 이 FMD 프로토콜을 적용하지 않습니다.')
                continue
            m = measure_annotation(ann, frame)
            if m['mean_diameter_mm'] is None:
                out['issues'].append('승인된 프레임 모두에 유효한 길이 보정과 내강 경계가 필요합니다.')
                continue
            phase = ann.get('phase')
            t = frame['timestamp_seconds']
            if phase == 'baseline':
                inflation = ann.get('cuff_inflation_video_seconds')
                if inflation is None or not -30 <= t - inflation <= 0:
                    out['issues'].append('기준 프레임은 커프 팽창 전 30초 구간인지 확인해야 합니다.')
                    continue
                out['baseline_frames'].append({'frame_id': frame['id'], 'diameter_mm': m['mean_diameter_mm']})
            elif phase == 'post_release':
                release = ann.get('cuff_release_video_seconds')
                if release is None or not 60 <= t - release <= 75:
                    continue
                out['peak_frames'].append({'frame_id': frame['id'], 'diameter_mm': m['mean_diameter_mm']})
        if len(out['baseline_frames']) != 5 or len(out['peak_frames']) != 5:
            out['issues'].append('논문 방식 계산에는 승인된 기준 5장과 커프 해제 후 60~75초 구간 5장을 지정하세요. 다른 프레임은 구간 미지정으로 두세요.')
        out['issues'] = list(dict.fromkeys(out['issues']))
        if not out['issues']:
            out['baseline_mm'] = statistics.mean(f['diameter_mm'] for f in out['baseline_frames'])
            out['peak_mm'] = max(f['diameter_mm'] for f in out['peak_frames'])
            out['absolute_fmd_mm'] = out['peak_mm'] - out['baseline_mm']
            out['fmd_percent'] = 100 * out['absolute_fmd_mm'] / out['baseline_mm']
        results.append(out)
    return {'sessions': results, 'note': '수동 경계와 입력한 프로토콜 정보를 이용한 연구용 참고 계산. 내피 기능의 임상 판정이 아닙니다.'}
