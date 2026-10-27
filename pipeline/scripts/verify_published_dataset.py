"""Check the published snapshot before using its images, labels or masks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
report=json.loads((REPO/'research/dataset_integrity.json').read_text(encoding='utf-8'))
for row in report['frames']:
    for kind in ['image','mask','annotation']:
        p=REPO/row[kind]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row[kind+'_sha256'],p
    mask=np.array(Image.open(REPO/row['mask']))
    assert mask.shape==(448,640) and set(np.unique(mask)).issubset({0,1}),row['frame_id']
    ann=json.loads((REPO/row['annotation']).read_text(encoding='utf-8'))
    assert ann['status']=='approved',row['frame_id']
assert len(report['frames'])==215
print('Verified: 215 original images, 215 approved polygons, 215 binary masks (0/1).')
