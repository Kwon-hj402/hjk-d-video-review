"""Checks group leakage, polygon coordinate export, and FMD gating on fixtures."""
import json
import tempfile
from pathlib import Path

from PIL import Image
from export_dataset import export
from measurement import measure_annotation, summarize_fmd


def main():
    frame={'id':'sample','timestamp_seconds':5,'width':640,'height':448}
    ann={'status':'approved','objects':[{'class_id':0,'points':[[100,180],[500,180],[500,240],[100,240]]}],
         'mm_per_pixel':.05,'patient_id':'P01','session_id':'S01','anatomy':'brachial'}
    m=measure_annotation(ann,frame)
    assert abs(m['mean_diameter_px']-60)<1e-8 and abs(m['mean_diameter_mm']-3)<1e-8
    assert summarize_fmd([(frame,ann)])['sessions'][0]['fmd_percent'] is None
    rows=[]
    for i in range(5):
        baseline=dict(ann,phase='baseline',cuff_inflation_video_seconds=30)
        peak=dict(ann,phase='post_release',cuff_release_video_seconds=0,
                  objects=[{'class_id':0,'points':[[100,180],[500,180],[500,246],[100,246]]}])
        rows.extend([(dict(frame,id=f'b{i}',timestamp_seconds=i+5),baseline),
                     (dict(frame,id=f'p{i}',timestamp_seconds=60+i),peak)])
    fmd=summarize_fmd(rows)['sessions'][0]
    assert abs(fmd['fmd_percent']-10)<1e-8
    bad=[(f,dict(a,anatomy='carotid')) for f,a in rows]
    assert summarize_fmd(bad)['sessions'][0]['fmd_percent'] is None
    with tempfile.TemporaryDirectory(prefix='fmd_pipeline_') as td:
        root=Path(td)
        (root/'data/annotations').mkdir(parents=True)
        (root/'data/all_frames').mkdir()
        frames=[]
        for patient in range(6):
            for i in range(2):
                fid=f'p{patient}_f{i}'
                fr=dict(frame,id=fid,video_id=f'v{patient}',frame_index=i,patient_id='unassigned',
                        image_path=f'data/all_frames/{fid}.png')
                frames.append(fr)
                Image.new('RGB',(640,448),(30,30,30)).save(root/fr['image_path'])
                label=dict(ann,frame_id=fid,patient_id=f'P{patient}')
                (root/'data/annotations'/f'{fid}.json').write_text(json.dumps(label),encoding='utf-8')
        (root/'data/manifest.json').write_text(json.dumps({'frames':frames}),encoding='utf-8')
        result=export(root)
        report=json.loads((root/'data/export/current/split_report.json').read_text())
        groups=[set(report['patient_groups'][s]) for s in ('train','val','test')]
        assert all(not a.intersection(b) for i,a in enumerate(groups) for b in groups[i+1:])
        for fr in report['frames']:
            label=(root/'data/export/current/labels'/fr['split']/(fr['frame_id']+'.txt')).read_text().split()
            assert label[0]=='0' and abs(float(label[1])-100/640)<1e-7
            mask=Image.open(root/'data/export/current/masks'/fr['split']/(fr['frame_id']+'.png'))
            assert mask.getpixel((200,200))==1 and mask.getpixel((200,100))==0
        version_cfg=(Path(result['versioned_output'])/'dataset.yaml').read_text()
        assert result['versioned_output'] in version_cfg and '/current' not in version_cfg
        for p in (root/'data/annotations').glob('*.json'):
            a=json.loads(p.read_text());a['patient_id']='unassigned';p.write_text(json.dumps(a))
        try: export(root)
        except ValueError: pass
        else: raise AssertionError('Unassigned patients must not enter strict split')
    print('PASS: mask coordinates, patient leakage prevention, immutable exports, missing metadata and anatomy FMD gates, 10% known FMD fixture')


if __name__=='__main__':
    main()
