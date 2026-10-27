"""Re-evaluate the shipped checkpoint against approved original-resolution masks."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from _runtime import ROOT, configure


def overlap(gt, pred):
    intersection=int(np.count_nonzero(gt & pred))
    gp,pp=int(gt.sum()),int(pred.sum())
    union=gp+pp-intersection
    return dict(gt_pixels=gp,pred_pixels=pp,intersection_pixels=intersection,
                dice=2*intersection/(gp+pp) if gp+pp else None,
                iou=intersection/union if union else None,both_empty=gp+pp==0)


def main():
    parser=argparse.ArgumentParser(description='승인 마스크와 모델 예측의 원본 해상도 Dice/IoU 평가')
    parser.add_argument('--weights',type=Path,default=ROOT/'models/best.pt')
    parser.add_argument('--data',type=Path,default=ROOT/'data/export/current')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--splits',nargs='+',choices=['val','test'],default=['val','test'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    configure()
    from ultralytics import YOLO
    model=YOLO(str(args.weights),task='segment')
    report=json.loads((args.data/'split_report.json').read_text(encoding='utf-8'))
    rows=[]
    for frame in report['frames']:
        if frame['split'] not in args.splits: continue
        fid,kind=frame['frame_id'],frame['split']
        gt=np.array(Image.open(args.data/'masks'/kind/(fid+'.png')))>0
        image=args.data/'images'/kind/(fid+'.png')
        result=model.predict(str(image),device=args.device,imgsz=640,conf=.25,retina_masks=True,verbose=False)[0]
        pred=np.zeros_like(gt)
        if result.masks is not None:
            for mask in result.masks.data.cpu().numpy():
                if mask.shape != gt.shape: raise RuntimeError('Prediction is not at the original resolution')
                pred |= mask>.5
        row=dict(frame_id=fid,split=kind,negative=frame['negative'],**overlap(gt,pred))
        rows.append(row)
        path=args.output/'masks'/(fid+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pred.astype(np.uint8)*255).save(path)
    summary={}
    for kind in args.splits:
        values=[r for r in rows if r['split']==kind]
        positive=[r for r in values if not r['negative']]
        negative=[r for r in values if r['negative']]
        summary[kind]=dict(frames=len(values),positive_frames=len(positive),negative_frames=len(negative),
                          positive_mean_dice=float(np.mean([r['dice'] for r in positive])) if positive else None,
                          positive_mean_iou=float(np.mean([r['iou'] for r in positive])) if positive else None,
                          negative_frames_without_predictions=sum(r['pred_pixels']==0 for r in negative))
    with (args.output/'per_frame_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/'metrics.json').write_text(json.dumps({'confidence':.25,'empty_mask_policy':'both-empty Dice and IoU are null',
        'patient_independent_validation':False,'splits':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
