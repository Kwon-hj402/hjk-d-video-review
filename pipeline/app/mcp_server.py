"""MCP tools share the local labeler's API; proposals stay unapproved drafts."""
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP, Image

BASE = 'http://127.0.0.1:8765'
mcp = FastMCP('FMD Korean Labeler')


def api(path, body=None):
    payload=None if body is None else json.dumps(body,ensure_ascii=False).encode('utf-8')
    req=Request(BASE+path,data=payload,headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=180) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        detail=e.read().decode('utf-8',errors='replace')
        raise ValueError(f'작업대 요청 실패 ({e.code}): {detail}') from e
    except URLError as e:
        raise ValueError('FMD 작업대가 실행되지 않았습니다. 레이블링_시작.cmd를 실행하세요.') from e


@mcp.tool()
def list_frames(selected: bool=True, video_id: str='', status: str='', offset: int=0, limit: int=20) -> dict:
    """선별 400장 또는 전체 프레임을 찾아 원본 프레임 ID와 레이블 상태를 반환합니다."""
    data=api('/api/frames?'+urlencode({'selected':str(selected).lower()}))
    frames=data if isinstance(data,list) else data.get('frames',[])
    if video_id: frames=[f for f in frames if f['video_id']==video_id]
    if status: frames=[f for f in frames if f.get('status',f.get('annotation_status','unlabeled'))==status]
    return {'total':len(frames),'frames':frames[max(0,offset):max(0,offset)+min(100,max(1,limit))]}


@mcp.tool()
def inspect_frame(frame_id: str) -> Image:
    """허용된 원본 프레임 이미지를 표시합니다. 이 이미지를 보고 경계를 검토하세요."""
    with urlopen(BASE+'/api/image/'+quote(frame_id,safe=''),timeout=30) as response:
        return Image(data=response.read(),format='png')


@mcp.tool()
def get_annotation(frame_id: str) -> dict:
    """원본 픽셀 좌표의 내강 다각형, 승인 상태, 촬영 정보를 읽습니다."""
    return api('/api/annotation/'+quote(frame_id,safe=''))


@mcp.tool()
def save_draft(frame_id: str, objects: list[dict], notes: str='AI가 작성한 초안 · 사용자 확인 필요') -> dict:
    """검토용 초안을 저장합니다. 사람이 승인한 레이블은 이 도구로 덮어쓰지 않습니다."""
    ann=get_annotation(frame_id)
    if 'annotation' in ann: ann=ann['annotation']
    if ann.get('status')=='approved':
        raise ValueError('이미 승인된 레이블입니다. 사용자에게 UI에서 초안으로 전환하도록 요청하세요.')
    ann.update({'frame_id':frame_id,'status':'draft','objects':objects,'notes':notes})
    return api('/api/annotation/'+quote(frame_id,safe=''),ann)


@mcp.tool()
def suggest_boundary(frame_id: str, box: list[float]) -> dict:
    """혈관 내강을 감싼 원본좌표 상자 [x1,y1,x2,y2]로 SAM2 경계 초안을 제안합니다. 자동 승인하지 않습니다."""
    return api('/api/assist/'+quote(frame_id,safe=''),{'box':box})


@mcp.tool()
def measure_frame(frame_id: str) -> dict:
    """내강 경계의 수직 직경과 보정값이 있는 경우 mm 값을 읽습니다."""
    return api('/api/measurement/'+quote(frame_id,safe=''))


@mcp.tool()
def readiness() -> dict:
    """프레임 수, 승인 수, 환경과 학습 준비 상태를 확인합니다."""
    return api('/api/health')


@mcp.tool()
def export_approved(mode: str='patient') -> dict:
    """사람이 승인한 레이블만 COCO, 픽셀 마스크, YOLO로 내보냅니다. 기본은 대상자 단위 분리입니다."""
    if mode not in ('patient','video-exploratory'):
        raise ValueError('mode는 patient 또는 video-exploratory여야 합니다.')
    return api('/api/export',{'mode':mode})


if __name__=='__main__':
    mcp.run(transport='stdio')
