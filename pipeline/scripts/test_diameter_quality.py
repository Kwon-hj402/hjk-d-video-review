"""Focused checks for the optional spatial-support gate; patient inputs read only."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

from diameter_quality import apply_spatial_support_gate, SPATIAL_SUPPORT_GATE_VERSION
from measurement import measure_mask
from test_measurement_geometry import rectangle, curved_constant_width


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT/'reports/video_predictions/all_videos_fresh_20261008T142258Z'
EVIDENCE = {}


def fixture(span=35, axis=(1, 0)):
    return {'status':'ok', 'mean_diameter_px':10., 'mean_diameter_mm':.5,
            'pca_longitudinal_extent_px':span, 'principal_axis':list(axis),
            'scanlines':[{'x1':10.,'y1':10.,'x2':10.,'y2':20.,'diameter_px':10.}],
            'scanline_count':1, 'sample_count':2, 'excluded_scanline_count':1,
            'coverage':.5, 'measured_range':{'start_t':0.,'end_t':35.},
            'warnings':[], 'centerline':[[10.,15.]]}


class SpatialSupportTests(unittest.TestCase):
    def test_threshold_equality_and_rejection_diagnostics(self):
        accepted = apply_spatial_support_gate(fixture(35), [0,0,100,80])
        self.assertTrue(accepted['spatial_support_gate']['passed'])
        self.assertEqual(accepted['mean_diameter_px'],10)
        rejected = apply_spatial_support_gate(fixture(34.999), [0,0,100,80])
        self.assertEqual(rejected['status'],'unmeasurable')
        self.assertIsNone(rejected['mean_diameter_px'])
        self.assertIsNone(rejected['mean_diameter_mm'])
        self.assertIsNone(rejected['measured_range'])
        self.assertEqual(rejected['scanlines'],[])
        self.assertEqual(rejected['scanline_count'],0)
        self.assertEqual(rejected['coverage'],0)
        self.assertEqual(rejected['excluded_scanline_count'],2)
        self.assertEqual(rejected['quality_gate_rejected_scanline_count'],1)
        self.assertEqual(len(rejected['candidate_scanlines']),1)
        self.assertEqual(rejected['spatial_support_gate']['candidate_mean_diameter_px'],10)
        self.assertEqual(rejected['message'],'검출한 혈관 길이가 부족하여 측정하지 않습니다.')
        json.dumps(rejected,allow_nan=False)

    def test_no_input_mutation_or_nested_alias(self):
        original=fixture(20); before=deepcopy(original)
        after=apply_spatial_support_gate(original,[0,0,100,80])
        self.assertEqual(original,before)
        after['candidate_scanlines'][0]['x1']=999
        after['centerline'][0][0]=999
        after['warnings'].append('changed')
        self.assertEqual(original,before)

    def test_physical_axis_projection_and_90_degree_rotation(self):
        horizontal=apply_spatial_support_gate(fixture(330,(1,0)),[94,40,545,382])
        vertical=apply_spatial_support_gate(fixture(330,(0,1)),[40,94,382,545])
        mirrored=apply_spatial_support_gate(fixture(330,(-3,0)),[94,40,545,382])
        ratios=[m['spatial_support_gate']['axis_coverage'] for m in (horizontal,vertical,mirrored)]
        self.assertAlmostEqual(ratios[0],ratios[1],places=12)
        self.assertAlmostEqual(ratios[0],ratios[2],places=12)
        theta=math.radians(37); axis=(math.cos(theta),math.sin(theta))
        corners=[(0,0),(100,0),(0,80),(100,80)]
        projections=[x*axis[0]+y*axis[1] for x,y in corners]
        expected=max(projections)-min(projections)
        gate=apply_spatial_support_gate(fixture(80,axis),[0,0,100,80])['spatial_support_gate']
        self.assertAlmostEqual(gate['projected_roi_span_px'],expected,places=12)

    def test_heldout_full_rotated_vessels(self):
        results=[]
        for angle in (-72,-49,-8,11,31,56,78):
            m=measure_mask(rectangle(diameter=64,angle=angle,length=320))
            out=apply_spatial_support_gate(m,m['roi'])
            self.assertEqual(out['status'],'ok')
            self.assertEqual(out['scanlines'],m['scanlines'])
            self.assertEqual(out['mean_diameter_px'],m['mean_diameter_px'])
            self.assertAlmostEqual(out['mean_diameter_px'],64,delta=.8)
            results.append({'angle':angle,'Dpx':out['mean_diameter_px'],
                            'axis_coverage':out['spatial_support_gate']['axis_coverage']})
        EVIDENCE['heldout_rotations']=results

    def test_curvature_uniform_dilation_and_short_fragment(self):
        results=[]
        for name,mask in [('width60',rectangle(60,25)),('width72',rectangle(72,25)),
                          ('curved50',curved_constant_width(50)),
                          ('short_fragment',rectangle(25,0,110))]:
            m=measure_mask(mask); out=apply_spatial_support_gate(m,m['roi'])
            self.assertEqual(m['status'],'ok')
            if name=='short_fragment':
                self.assertIsNone(out['mean_diameter_px'])
                self.assertFalse(out['spatial_support_gate']['passed'])
            else:
                self.assertEqual(out['mean_diameter_px'],m['mean_diameter_px'])
            results.append({'name':name,'Dpx':out['mean_diameter_px'],
                            'axis_coverage':out['spatial_support_gate']['axis_coverage']})
        self.assertAlmostEqual(results[1]['Dpx']/results[0]['Dpx'],1.2,delta=.015)
        EVIDENCE['curvature_dilation_fragment']=results

    def test_missing_negative_and_bad_geometry_are_not_promoted(self):
        for status in ('unmeasurable','negative','invalid',None):
            m=fixture(80); m.update(status=status,mean_diameter_px=None,mean_diameter_mm=None)
            out=apply_spatial_support_gate(m,[0,0,100,80])
            self.assertEqual(out['status'],status)
            self.assertIsNone(out['mean_diameter_px'])
        for omitted in ('principal_axis','pca_longitudinal_extent_px'):
            m=fixture(80); del m[omitted]
            out=apply_spatial_support_gate(m,[0,0,100,80])
            self.assertEqual(out['status'],'unmeasurable')
            self.assertFalse(out['spatial_support_gate']['evaluated'])
            self.assertIsNone(out['mean_diameter_px'])
        for span,axis in [(-1,(1,0)),(float('nan'),(1,0)),(80,(0,0))]:
            out=apply_spatial_support_gate(fixture(span,axis),[0,0,100,80])
            self.assertIsNone(out['mean_diameter_px'])
            json.dumps(out,allow_nan=False)
        out=apply_spatial_support_gate(fixture(80),None)
        self.assertIsNone(out['mean_diameter_px'])

    def test_configuration_validation_and_repeat_check(self):
        for minimum in (-.01,1.01,float('nan'),None):
            with self.assertRaises(ValueError):
                apply_spatial_support_gate(fixture(),[0,0,100,80],minimum)
        for roi in ([0,0,0,80],[0,0,100],[0,0,float('inf'),80],'bad'):
            with self.assertRaises(ValueError):
                apply_spatial_support_gate(fixture(),roi)
        first=apply_spatial_support_gate(fixture(20),[0,0,100,80])
        again=apply_spatial_support_gate(first,[0,0,100,80])
        self.assertIsNone(again['mean_diameter_px'])
        self.assertEqual(again['excluded_scanline_count'],first['excluded_scanline_count'])
        self.assertEqual(again['spatial_support_gate']['candidate_mean_diameter_px'],10)

    def test_real_batch_and_offending_frames(self):
        self.assertTrue(BATCH.is_dir(),'Expected read-only batch fixture missing')
        totals={'processed':0,'prior_measured':0,'retained_measured':0,'unmeasurable':0,'newly_rejected':0}
        per_video=[]; targets=[]
        target_keys={('0004',963):False,('0006',1590):False,('0004',197):True,('0006',1895):True}
        for directory in sorted(BATCH.glob('EBL*')):
            counts={'video':directory.name[-4:],'processed':0,'prior_measured':0,'retained_measured':0,'unmeasurable':0,'newly_rejected':0}
            with (directory/'diameter_geometry.jsonl').open(encoding='utf-8') as source:
                for line in source:
                    record=json.loads(line); m=record['measurement']
                    out=apply_spatial_support_gate(m,m.get('roi'))
                    before=m.get('mean_diameter_px') is not None
                    after=out.get('mean_diameter_px') is not None
                    self.assertFalse(after and not before,'Bad geometry was promoted')
                    if after:
                        self.assertEqual(out['mean_diameter_px'],m['mean_diameter_px'])
                        self.assertEqual(out['scanlines'],m['scanlines'])
                    counts['processed']+=1; counts['prior_measured']+=before
                    counts['retained_measured']+=after; counts['unmeasurable']+=not after
                    counts['newly_rejected']+=before and not after
                    key=(counts['video'],record['frame_index'])
                    if key in target_keys:
                        self.assertEqual(after,target_keys[key])
                        targets.append({'video':key[0],'frame':key[1],'retained':after,
                                        'spatial_support_gate':out['spatial_support_gate']})
            per_video.append(counts)
            for name in totals:
                totals[name]+=counts[name]
        self.assertEqual(totals,{'processed':15133,'prior_measured':12363,'retained_measured':12276,
                                'unmeasurable':2857,'newly_rejected':87})
        self.assertEqual([v['newly_rejected'] for v in per_video],[3,26,11,47,0,0,0])
        self.assertEqual(len(targets),4)
        EVIDENCE.update(batch_totals=totals,per_video=per_video,real_critical_frames=targets)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpatialSupportTests))
    report={'version':SPATIAL_SUPPORT_GATE_VERSION,'minimum_axis_coverage':.35,
            'tests_run':run.testsRun,'failures':len(run.failures),'errors':len(run.errors),
            'successful':run.wasSuccessful(),'no_temporal_filter_or_diameter_bounds':True,
            'source_patient_files_modified':False,'evidence':EVIDENCE}
    (ROOT/'reports/diameter_quality_verification.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    raise SystemExit(0 if run.wasSuccessful() else 1)
