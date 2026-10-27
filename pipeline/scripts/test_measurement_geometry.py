"""Focused independent geometric fixtures; no FMD or temporal calculations."""
import json
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from measurement import measure_annotation, measure_mask

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {}


def rectangle(diameter=60, angle=0, length=330):
    yy, xx = np.indices((448, 640), dtype=float)
    radians = math.radians(angle)
    t = (xx-319.5)*math.cos(radians)+(yy-210.5)*math.sin(radians)
    n = -(xx-319.5)*math.sin(radians)+(yy-210.5)*math.cos(radians)
    return ((np.abs(t) < length/2)&(np.abs(n) < diameter/2)).astype(np.uint8)


def polygon_mask(points):
    # Analytic ray crossing at integer pixel centers, independent of OpenCV
    # inclusive raster fill. This fixture preserves the specified offset walls.
    points = np.asarray(points)
    yy, xx = np.indices((448, 640), dtype=float)
    inside = np.zeros((448, 640), dtype=bool)
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        if a[1] == b[1]:
            continue
        cross = (a[1] > yy) != (b[1] > yy)
        inside ^= cross & (xx < a[0]+(yy-a[1])*(b[0]-a[0])/(b[1]-a[1]))
    return inside.astype(np.uint8)


def curved_constant_width(width=50):
    x = np.linspace(140, 500, 241)
    y = 218 + 35*np.sin(np.pi*(x-140)/360)
    slope = 35*np.pi/360*np.cos(np.pi*(x-140)/360)
    norm = np.column_stack((-slope, np.ones_like(x)))
    norm /= np.linalg.norm(norm, axis=1)[:, None]
    centers = np.column_stack((x, y))
    upper, lower = centers-width/2*norm, centers+width/2*norm
    return polygon_mask(np.vstack((upper, lower[::-1])))


def record(name, mask, **kwargs):
    before = mask.copy()
    result = measure_mask(mask, **kwargs)
    assert np.array_equal(mask, before, equal_nan=True), 'Input mask was mutated'
    json.dumps(result, allow_nan=False)
    RESULTS[name] = result
    return result


class DiameterGeometryTests(unittest.TestCase):
    def assert_measured(self, result, expected, delta=1):
        self.assertEqual(result['status'], 'ok', result['message'])
        self.assertAlmostEqual(result['mean_diameter_px'], expected, delta=delta)
        self.assertGreaterEqual(result['coverage'], .45)
        self.assertEqual(result['sample_count'], result['scanline_count']+result['excluded_scanline_count'])
        for line in result['scanlines']:
            dx, dy = line['x2']-line['x1'], line['y2']-line['y1']
            self.assertAlmostEqual(math.hypot(dx, dy), line['diameter_px'], places=8)
            self.assertAlmostEqual(dx*line['tangent_x']+dy*line['tangent_y'], 0, places=8)

    def test_parallel_walls_and_calibration(self):
        result = record('parallel_60', rectangle(), mm_per_pixel=.078003120124805)
        self.assert_measured(result, 60, delta=.05)
        self.assertAlmostEqual(result['mean_diameter_mm'], 60*.078003120124805, delta=.005)
        requested = result['requested_range']
        self.assertEqual(requested['end_trim_fraction_each'], .15)
        self.assertAlmostEqual(requested['end_t']-requested['start_t'], .7*result['pca_longitudinal_extent_px'], places=7)

    def test_rotation_invariance_and_local_normal(self):
        diameters = []
        for angle in (-60, -35, -15, 0, 17, 40, 65, 89):
            result = record(f'rotated_{angle}', rectangle(angle=angle))
            self.assert_measured(result, 60, delta=.8)
            diameters.append(result['mean_diameter_px'])
        self.assertLess(max(diameters)-min(diameters), 1)

    def test_uniform_dilation_is_preserved(self):
        baseline = record('dilation_60', rectangle(diameter=60, angle=25))
        dilation = record('dilation_72', rectangle(diameter=72, angle=25))
        self.assert_measured(baseline, 60, delta=.8)
        self.assert_measured(dilation, 72, delta=.8)
        self.assertAlmostEqual(dilation['mean_diameter_px']/baseline['mean_diameter_px'], 1.2, delta=.015)
        self.assertGreater(dilation['coverage'], .9)

    def test_curved_constant_width(self):
        result = record('curved_50', curved_constant_width())
        self.assert_measured(result, 50, delta=.8)
        self.assertGreater(result['coverage'], .9)
        slopes = [line['tangent_y'] for line in result['scanlines']]
        self.assertGreater(max(slopes)-min(slopes), .25)

    def test_branch_and_multiple_intersection_are_excluded(self):
        mask = rectangle(length=360)
        cv2.line(mask, (285, 182), (355, 125), 1, 19)
        result = record('branch', mask)
        self.assert_measured(result, 60, delta=.9)
        self.assertGreater(result['excluded_scanline_count'], 5)
        reasons = {line['reason'] for line in result['rejected_scanlines']}
        self.assertTrue(reasons & {'multiple_intersections', 'spatial_width_outlier', 'normal_width_mad_outlier', 'rapid_boundary_change'})

    def test_narrow_end_caps_do_not_set_diameter(self):
        yy, xx = np.indices((448, 640), dtype=float)
        t, n = xx-319.5, yy-210.5
        halfwidth = np.where(np.abs(t) <= 125, 30, np.maximum(0, 30*(175-np.abs(t))/50))
        mask = ((np.abs(t) < 175)&(np.abs(n) < halfwidth)).astype(np.uint8)
        result = record('narrow_caps', mask)
        self.assert_measured(result, 60, delta=.05)
        self.assertTrue(all(195 < line['center_x'] < 445 for line in result['scanlines']))

    def test_noise_components_and_small_holes(self):
        mask = rectangle()
        for center in ((120, 130), (510, 310), (230, 100)):
            cv2.circle(mask, center, 3, 1, -1)
        mask[205:208, 315:318] = 0
        result = record('noise_hole', mask)
        self.assert_measured(result, 60, delta=.05)
        self.assertEqual(result['ignored_component_count'], 3)
        self.assertEqual(result['filled_hole_count'], 1)
        self.assertEqual(result['filled_hole_pixels'], 9)

    def test_degenerate_missing_and_roi_clipping(self):
        empty = np.zeros((448, 640), np.uint8)
        thin = rectangle(diameter=2)
        disk = empty.copy(); cv2.circle(disk, (320, 210), 45, 1, -1)
        clipped = empty.copy(); clipped[350:411, 120:521] = 1
        for name, mask in [('empty', empty), ('thin', thin), ('disk', disk), ('roi_clipped', clipped)]:
            result = record(name, mask)
            self.assertIsNone(result['mean_diameter_px'])
            self.assertNotEqual(result['status'], 'ok')
        invalid = np.full((448, 640), np.nan)
        result = record('invalid_nan', invalid)
        self.assertEqual(result['status'], 'invalid')

    def test_polygon_compatibility_multiple_and_negative(self):
        frame = {'width':640, 'height':448}
        ann = {'status':'approved', 'mm_per_pixel':.05, 'objects':[
            {'class_id':0, 'points':[[100,180],[500,180],[500,240],[100,240]]},
            {'class_id':0, 'points':[[110,110],[118,110],[115,116]]},
        ]}
        result = measure_annotation(ann, frame)
        RESULTS['annotation_multiple'] = result
        self.assert_measured(result, 60, delta=.0001)
        self.assertAlmostEqual(result['mean_diameter_mm'], 3, places=7)
        self.assertEqual(result['annotation_object_count'], 2)
        self.assertIsNone(measure_annotation({}, frame)['mean_diameter_px'])
        self.assertIsNone(measure_annotation(dict(ann,status='negative'),frame)['mean_diameter_px'])
        crossed = dict(ann, objects=[{'class_id':0,'points':[[120,180],[500,240],[120,240],[500,180]]}])
        json.dumps(measure_annotation(crossed,frame), allow_nan=False)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiameterGeometryTests)
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'method':'local_normal_mask_geometry', 'tests_run':run.testsRun,
              'failures':len(run.failures), 'errors':len(run.errors), 'successful':run.wasSuccessful(),
              'no_temporal_filtering_or_fmd_calculation':True, 'fixtures':RESULTS}
    (ROOT/'reports/diameter_geometry_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    raise SystemExit(0 if run.wasSuccessful() else 1)
