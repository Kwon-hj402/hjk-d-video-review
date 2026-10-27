"""Export approved polygons to masks/COCO/YOLO with leakage-safe group splits."""
import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def effective_annotation(frame, ann, video_metadata):
    meta = video_metadata.get(frame['video_id'], {})
    result = dict(meta)
    result.update(ann)
    if meta.get('updated_at', '') > ann.get('updated_at', ''):
        result.update({k:v for k,v in meta.items() if k != 'updated_at'})
    result.setdefault('patient_id', frame.get('patient_id', 'unassigned'))
    return result


def polygon_area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1]))) / 2


def validate_objects(objects, width, height):
    clean = []
    for obj in objects:
        pts = obj.get('points', [])
        if obj.get('class_id', 0) != 0 or len(pts) < 3:
            raise ValueError('내강 클래스의 3점 이상 다각형이 필요합니다.')
        if any(len(p)!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p) for p in pts):
            raise ValueError('다각형 좌표가 유효하지 않습니다.')
        if any(not (0<=x<=width and 0<=y<=height) for x,y in pts):
            raise ValueError('원본 영상 밖의 좌표가 있습니다.')
        if polygon_area(pts) < 4:
            raise ValueError('내강 다각형 면적이 너무 작습니다.')
        clean.append({'class_id':0, 'label':'lumen', 'points':pts})
    if not clean:
        raise ValueError('승인된 프레임에 내강 경계가 없습니다.')
    return clean


def assign_groups(rows, mode, seed=42):
    groups = defaultdict(list)
    key = 'patient_id' if mode=='patient' else 'video_id'
    for frame, ann in rows:
        group = ann.get('patient_id','unassigned') if key=='patient_id' else frame['video_id']
        if mode=='patient' and (not str(group).strip() or str(group).lower() in ('unassigned','unknown','미확인','미지정')):
            raise ValueError('대상자 ID가 미지정입니다. 영상별 촬영 정보에 실제 대상자를 지정하세요. 같은 대상자의 영상은 같은 ID여야 합니다.')
        groups[str(group)].append((frame,ann))
    if len(groups)<3:
        raise ValueError('훈련·검증·시험을 분리하려면 최소 3개의 서로 다른 대상자 그룹이 필요합니다.' if mode=='patient' else '영상 그룹 분리에는 최소 3개의 승인된 원본 영상이 필요합니다.')
    keys=sorted(groups)
    random.Random(seed).shuffle(keys)
    ntest=max(1, round(len(keys)*.1))
    nval=max(1, round(len(keys)*.1))
    mapping={k:'test' for k in keys[:ntest]}
    mapping.update({k:'val' for k in keys[ntest:ntest+nval]})
    mapping.update({k:'train' for k in keys[ntest+nval:]})
    return groups,mapping


def export(root=ROOT, mode='patient', seed=42, labels_only=False):
    manifest=json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
    meta_path=root/'data/video_metadata.json'
    video_meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    # UI stores direct map; wrapper tolerated for portability.
    video_meta=video_meta.get('videos',video_meta)
    rows=[]
    annotation_dir=root/'data/annotations'
    errors=[]
    for frame in manifest['frames']:
        p=annotation_dir/(frame['id']+'.json')
        if not p.exists():
            continue
        ann=effective_annotation(frame,json.loads(p.read_text(encoding='utf-8')),video_meta)
        if ann.get('status')!='approved':
            continue
        try:
            if ann.get('negative') and not ann.get('objects'):
                ann['objects']=[]
            else:
                ann['objects']=validate_objects(ann.get('objects',[]),frame['width'],frame['height'])
        except ValueError as e:
            errors.append(frame['id']+': '+str(e))
        rows.append((frame,ann))
    if errors:
        raise ValueError('\n'.join(errors[:20]))
    if not rows:
        raise ValueError('승인된 내강 레이블이 아직 없습니다. 초안을 확인하고 승인한 뒤 내보내세요.')
    if not any(ann['objects'] for _,ann in rows):
        raise ValueError('내강 양성 레이블이 최소 한 장 필요합니다. 내강 없음 자료만으로 segmentation을 학습할 수 없습니다.')
    if mode == 'patient':
        video_patients = defaultdict(set)
        for frame,ann in rows:
            video_patients[frame['video_id']].add(str(ann.get('patient_id','unassigned')))
        if any(len(patients)>1 for patients in video_patients.values()):
            raise ValueError('같은 원본 영상에 서로 다른 대상자 ID가 지정되어 있습니다. 영상별 촬영 정보를 확인하세요.')
    groups,mapping=({}, {}) if labels_only else assign_groups(rows,mode,seed)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=root/'data/export'/stamp
    out.mkdir(parents=True)
    if annotation_dir.exists():
        shutil.copytree(annotation_dir,out/'annotation_backup')
    if meta_path.exists():
        shutil.copy2(meta_path,out/'video_metadata.json')
    coco={'info':{'description':'Approved lumen polygons in original pixel coordinates', 'date_created':stamp},
          'images':[], 'annotations':[], 'categories':[{'id':1,'name':'lumen'}]}
    split_report={'version':1, 'approved_only':True,'exploratory':mode!='patient',
                  'split_policy':'patient_group' if mode=='patient' else 'video_group_exploratory',
                  'seed':seed,'frame_counts':{},'patient_groups':{},'source_video_groups':{},
                  'frames':[], 'created_at':stamp, 'crop_policy':'original_full_frame',
                  'warnings':[]}
    if mode!='patient':
        split_report['warnings'].append('영상 단위 탐색 실험입니다. 여러 영상이 동일 대상자일 수 있어 대상자 독립 성능으로 해석하면 안 됩니다.')
    split_rows=defaultdict(list)
    ann_index=1
    for image_index,(frame,ann) in enumerate(rows,1):
        group=str(ann['patient_id']) if mode=='patient' else frame['video_id']
        split='unsplit' if labels_only else mapping[group]
        split_rows[split].append((frame,ann))
        image_dir=out/'images'/split
        label_dir=out/'labels'/split
        mask_dir=out/'masks'/split
        for d in (image_dir,label_dir,mask_dir):
            d.mkdir(parents=True,exist_ok=True)
        source=(root/frame['image_path']).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise ValueError('프레임 경로가 잘못되었습니다: '+frame['id'])
        image_path=image_dir/(frame['id']+'.png')
        try:
            image_path.hardlink_to(source)
        except OSError:
            shutil.copy2(source,image_path)
        width,height=frame['width'],frame['height']
        mask=Image.new('L',(width,height),0)
        draw=ImageDraw.Draw(mask)
        labels=[]
        for obj in ann['objects']:
            pts=obj['points']
            draw.polygon([tuple(p) for p in pts],fill=1)
            normalized=[v for x,y in pts for v in (x/width,y/height)]
            labels.append('0 '+' '.join(f'{v:.8f}' for v in normalized))
            xs,ys=[p[0] for p in pts],[p[1] for p in pts]
            coco['annotations'].append({'id':ann_index,'image_id':image_index,'category_id':1,
                'segmentation':[[v for p in pts for v in p]], 'area':polygon_area(pts),
                'bbox':[min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)],'iscrowd':0})
            ann_index+=1
        (label_dir/(frame['id']+'.txt')).write_text('\n'.join(labels)+'\n',encoding='utf-8')
        mask.save(mask_dir/(frame['id']+'.png'))
        coco['images'].append({'id':image_index,'file_name':image_path.relative_to(out).as_posix(),
                              'width':width,'height':height,'source_frame_id':frame['id']})
        split_report['frames'].append({'frame_id':frame['id'],'split':split,'patient_id':ann.get('patient_id'),
                                      'video_id':frame['video_id'],'frame_index':frame['frame_index'],
                                      'negative':bool(ann.get('negative',False))})
    for split,rs in split_rows.items():
        split_report['frame_counts'][split]=len(rs)
        split_report['patient_groups'][split]=sorted(set(str(a.get('patient_id','unassigned')) for f,a in rs))
        split_report['source_video_groups'][split]=sorted(set(f['video_id'] for f,a in rs))
    (out/'approved.coco.json').write_text(json.dumps(coco,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'split_report.json').write_text(json.dumps(split_report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not labels_only:
        cfg={'path':str(out.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'lumen'}}
        (out/'dataset.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    # A versioned export remains immutable; current is a lightweight copy with images hard-linked.
    current=root/'data/export/current'
    if current.exists():
        safe=current.resolve()
        if safe.parent!=(root/'data/export').resolve():
            raise ValueError('current 폴더 경로가 안전하지 않습니다.')
        shutil.rmtree(current)
    def link_or_copy(src,dst):
        if Path(src).suffix.lower() == '.png':
            try: Path(dst).hardlink_to(Path(src))
            except OSError: shutil.copy2(src,dst)
        else:
            shutil.copy2(src,dst)
        return dst
    shutil.copytree(out,current,copy_function=link_or_copy)
    if not labels_only:
        cfg['path']=str(current.resolve())
        (current/'dataset.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    return {'ok':True,'output':str(current),'versioned_output':str(out),'approved_frames':len(rows),
            'frame_counts':split_report['frame_counts'],'split_policy':split_report['split_policy'],
            'warnings':split_report['warnings'],'labels_only':labels_only}


def main():
    parser=argparse.ArgumentParser(description='승인된 레이블 내보내기')
    parser.add_argument('--mode',choices=['patient','video-exploratory'],default='patient')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--labels-only',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    try:
        print(json.dumps(export(args.root,args.mode,args.seed,args.labels_only),ensure_ascii=False))
    except (ValueError,FileNotFoundError) as e:
        print(json.dumps({'ok':False,'error':str(e)},ensure_ascii=False))
        sys.exit(2)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
