"""Local batch CLI for trained-model diameter videos; never computes FMD."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
KOREA = timezone(timedelta(hours=9), 'Asia/Seoul')
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv', '.webm'}
OUTPUT_MARKER = '.diameter_pipeline_output.json'


def _path_key(path):
    return os.path.normcase(str(path.resolve()))


def _within(path, directory):
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _produced_ancestor(path):
    directory = path if path.is_dir() else path.parent
    for parent in [directory, *directory.parents]:
        if (parent/OUTPUT_MARKER).is_file():
            return parent
    return None


def discover_sources(source_args, output=None):
    """Validate all paths, recursively collect videos, prune produced outputs."""
    inputs = []
    for value in source_args:
        raw = Path(value).expanduser()
        if not raw.exists():
            raise ValueError(f'입력 경로가 없습니다: {raw}')
        resolved = raw.resolve(strict=True)
        if not resolved.is_file() and not resolved.is_dir():
            raise ValueError(f'파일 또는 폴더를 지정하세요: {raw}')
        if resolved.is_file() and resolved.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f'지원하지 않는 영상 확장자입니다: {raw.name}')
        inputs.append(resolved)
    if not inputs:
        raise ValueError('원본 영상 파일이나 폴더를 지정하세요.')
    output = Path(output).resolve() if output is not None else None
    sources, seen, skipped_dirs, skipped_duplicates = [], set(), set(), []

    def excluded(path):
        if output is not None and _within(path, output):
            skipped_dirs.add(str(output)); return True
        prior = _produced_ancestor(path)
        if prior is not None:
            skipped_dirs.add(str(prior)); return True
        return False

    def add(path):
        resolved = path.resolve(strict=True)
        if excluded(resolved):
            return
        key = _path_key(resolved)
        if key in seen:
            skipped_duplicates.append(str(resolved))
        else:
            seen.add(key); sources.append(resolved)

    for entry in inputs:
        if excluded(entry):
            continue
        if entry.is_file():
            add(entry); continue
        for current, directories, filenames in os.walk(entry, followlinks=False):
            directory = Path(current)
            if excluded(directory):
                directories[:] = []; continue
            directories[:] = sorted(name for name in directories if not excluded(directory/name))
            for filename in sorted(filenames):
                path = directory/filename
                if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
                    add(path)
    if not sources:
        raise ValueError('처리할 원본 영상을 찾지 못했습니다. 이전 분석 결과 폴더는 자동으로 제외됩니다.')
    sources.sort(key=lambda path: _path_key(path))
    return dict(inputs=[str(path) for path in inputs], sources=sources,
                skipped_output_directories=sorted(skipped_dirs), skipped_duplicate_sources=skipped_duplicates)


def _configured_path(value, project_root):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    choices = [path] if path.is_absolute() else [project_root/path, project_root/'config'/path]
    return next((candidate.resolve() for candidate in choices if candidate.is_file()), None)


def resolve_weights(weights=None, project_root=ROOT):
    """Explicit checkpoint, or a dynamically resolved latest-inference config."""
    project_root = Path(project_root).resolve()
    config_path = project_root/'config/inference_latest.json'
    config, config_error = {}, None
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding='utf-8-sig'))
            if not isinstance(config, dict):
                raise ValueError('JSON object required')
        except (ValueError, OSError) as error:
            config_error = str(error)
            config = {}
    supported = ('weights', 'trained_weights', 'best_checkpoint')
    configured = {key: _configured_path(config.get(key), project_root) for key in supported}
    if weights is not None:
        checkpoint = Path(weights).expanduser()
        if not checkpoint.is_file():
            raise ValueError(f'모델 가중치 파일이 없습니다: {checkpoint}')
        checkpoint, selected_by = checkpoint.resolve(strict=True), '--weights'
        matching_config = any(candidate == checkpoint for candidate in configured.values() if candidate is not None)
    else:
        if config_error is not None:
            raise ValueError(f'최신 모델 설정을 읽을 수 없습니다: {config_path} ({config_error})')
        checkpoint = next((configured[key] for key in supported if configured[key] is not None), None)
        if checkpoint is None:
            raise ValueError('최신 학습 모델 설정이 없거나 가중치 파일을 찾을 수 없습니다. config/inference_latest.json을 준비하거나 --weights로 지정하세요.')
        selected_by, matching_config = str(config_path), True
    approved_info = {key: value for key, value in config.items()
                     if matching_config and ('snapshot' in key.lower() or 'approved' in key.lower())}
    model_info = dict(weights=str(checkpoint), selected_by=selected_by,
                      inference_config=str(config_path) if matching_config else None,
                      approved_snapshot_info=approved_info,
                      configuration=config if matching_config else None)
    return checkpoint, model_info


def video_directory_names(sources):
    grouped = {}
    for path in sources:
        grouped.setdefault(path.stem.casefold(), []).append(path)
    names = {}
    for group in grouped.values():
        for path in group:
            suffix = '__'+hashlib.sha256(_path_key(path).encode('utf-8')).hexdigest()[:8] if len(group) > 1 else ''
            names[path] = path.stem+suffix
    return names


def default_output_path():
    timestamp = datetime.now(KOREA).strftime('%Y%m%d_%H%M%S')
    parent = Path(r'C:\HJK\D_분석결과')
    base = parent/timestamp
    if not base.exists():
        return base
    for index in range(2, 10000):
        candidate = parent/f'{timestamp}_{index:02d}'
        if not candidate.exists():
            return candidate
    raise ValueError('새 결과 폴더 이름을 만들 수 없습니다.')


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def _relative_url(value, root):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root/path
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return quote(relative.as_posix(), safe='/')


def write_html(report, output):
    cards = []
    for entry in report['videos']:
        source = html.escape(Path(entry['source']).name)
        if entry['state'] not in ('complete', 'partial'):
            cards.append(f'<article class="card error"><h2>{source}</h2><span class="badge">처리 실패</span><p>{html.escape(entry.get("error", "처리 결과를 확인하세요."))}</p></article>')
            continue
        comparison = _relative_url(entry.get('comparison_video'), output)
        diameter = _relative_url(entry.get('diameter_video'), output)
        csv = _relative_url(entry.get('csv'), output)
        poster_path = Path(entry['output'])/'poster.jpg'
        poster = _relative_url(str(poster_path), output) if poster_path.is_file() else None
        poster_attribute = f' poster="{html.escape(poster,quote=True)}"' if poster else ''
        video = f'<video controls preload="metadata"{poster_attribute}><source src="{html.escape(comparison,quote=True)}" type="video/mp4">이 브라우저에서 영상을 재생할 수 없습니다.</video>' if comparison else '<p>비교 영상 경로를 확인하세요.</p>'
        links = ''.join(f'<a href="{html.escape(url,quote=True)}">{label}</a>' for label,url in [('원본 · 처리 비교 MP4',comparison),('D 표시 MP4',diameter),('프레임별 D CSV',csv)] if url)
        processed, measured, unavailable = entry.get('processed',0), entry.get('measured',0), entry.get('unmeasurable',0)
        state_label = '프레임 제한 미리보기' if entry['state'] == 'partial' else '완료'
        cards.append(f'<article class="card"><div class="card-top"><h2>{source}</h2><span class="badge">{state_label}</span></div>{video}<div class="counts"><span>처리 <b>{processed:,}</b></span><span>D 측정 <b>{measured:,}</b></span><span>측정 불가 <b>{unavailable:,}</b></span></div><nav>{links}</nav></article>')
    labels = {'running':'영상 분석 중','complete':'영상 분석 완료','partial':'프레임 제한 미리보기 완료','complete_with_errors':'일부 영상의 처리를 완료하지 못했습니다'}
    title = labels.get(report['state'], 'D 영상 분석')
    model_name = html.escape(Path(report['model']['weights']).name)
    document = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D 영상 분석 결과</title><style>
    :root{color-scheme:dark;font-family:"Malgun Gothic",system-ui,sans-serif;background:#111820;color:#e9f0f2}*{box-sizing:border-box}body{margin:0}main{max-width:1140px;margin:auto;padding:38px 24px 70px}header{margin-bottom:30px}h1{font-size:30px;margin:0 0 14px}p{line-height:1.65;color:#a8bbc3}.summary{display:flex;gap:18px;flex-wrap:wrap;color:#bfd2db;margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}.card{padding:20px;border:1px solid #2b414b;border-radius:15px;background:#18232c}.card-top{display:flex;align-items:flex-start;gap:12px;justify-content:space-between}h2{font-size:16px;overflow-wrap:anywhere;margin:0 0 16px}.badge{font-size:12px;color:#58dbc0;white-space:nowrap}video{display:block;width:100%;border-radius:8px;background:#000}.counts{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:#adc0c8;margin:18px 0}.counts b{color:#e9f0f2;margin-left:5px}nav{display:flex;gap:10px;flex-wrap:wrap}a{color:#72e2cc;text-decoration:none;border:1px solid #32524f;padding:9px 11px;border-radius:8px;font-size:12px}a:hover{background:#223c3a}.error{border-color:#71404a}.error .badge{color:#ffb3b3}footer{margin-top:28px;color:#92a9b3;font-size:13px;line-height:1.6}
    </style></head><body><main><header><h1>''' + html.escape(title) + '''</h1><p>왼쪽 원본과 오른쪽 처리 결과를 비교할 수 있습니다. 혈관 양끝과 분지·급변 경계를 제외하고, 중심선에 직각인 두 벽 사이의 평균 직경 D를 표시합니다.</p><div class="summary">'''+f'<span>원본 {report["source_count"]}개</span><span>완료 {report["completed_video_count"]}개</span><span>실패 {report["failed_video_count"]}개</span><span>모델 {model_name}</span>'+'''</div></header><section class="grid">'''+''.join(cards)+'''</section><footer>경계가 없거나 안정적인 측정 구간이 부족한 프레임에는 “측정 불가”를 표시합니다. 이 결과는 프레임별 D이며 FMD 계산은 포함하지 않습니다. 사용한 모델·배율·원본 경로와 처리 기록은 batch_report.json에서 확인할 수 있습니다.</footer></main></body></html>'''
    if report.get('partial_video_count'):
        document = document.replace(f'<span>실패 {report["failed_video_count"]}개</span>',
                                    f'<span>미리보기 {report["partial_video_count"]}개</span><span>실패 {report["failed_video_count"]}개</span>',1)
    (output/'index.html').write_text(document,encoding='utf-8')


def run_batch(source_args, output=None, weights=None, tick_mm=5.0, mm_per_pixel=None,
              roi=None, conf=.25, device='0', max_frames=None, renderer=None, project_root=ROOT):
    """Validate before writing, continue after video failures, save local reports."""
    output = Path(output).expanduser().resolve() if output is not None else default_output_path().resolve()
    discovery = discover_sources(source_args, output)
    checkpoint, model_info = resolve_weights(weights, project_root)
    if output.exists():
        raise ValueError(f'결과 폴더가 이미 존재합니다. 새 폴더를 지정하세요: {output}')
    numbers = [conf,tick_mm]+([mm_per_pixel] if mm_per_pixel is not None else [])
    if any(not math.isfinite(float(value)) for value in numbers) or not 0 < conf <= 1 or tick_mm <= 0 or (mm_per_pixel is not None and mm_per_pixel <= 0):
        raise ValueError('모델 신뢰도와 배율은 양수여야 하며 신뢰도는 1 이하여야 합니다.')
    if roi is not None and (len(roi) != 4 or any(not math.isfinite(float(value)) for value in roi) or roi[2] <= roi[0] or roi[3] <= roi[1]):
        raise ValueError('ROI는 x1,y1,x2,y2 순서의 유효한 네 좌표여야 합니다.')
    if max_frames is not None and max_frames < 1:
        raise ValueError('프레임 제한은 1 이상이어야 합니다.')
    if renderer is None:
        from video_pipeline import run_video
        renderer = run_video
    output.mkdir(parents=True,exist_ok=False)
    created = datetime.now(KOREA).isoformat(timespec='seconds')
    _write_json(output/OUTPUT_MARKER, dict(purpose='diameter_video_batch_output',schema_version=1,created_at=created,fmd_calculated=False))
    sources = discovery['sources']
    names = video_directory_names(sources)
    report = dict(schema_version=1,purpose='diameter_video_batch',state='running',created_at=created,
                  output=str(output),inputs=discovery['inputs'],sources=[str(path) for path in sources],
                  source_count=len(sources),model=model_info,approved_snapshot_info=model_info['approved_snapshot_info'],
                  fmd_calculated=False,completed_video_count=0,partial_video_count=0,failed_video_count=0,
                  processed_frame_count=0,measured_frame_count=0,unmeasurable_frame_count=0,videos=[],
                  skipped_output_directories=discovery['skipped_output_directories'],
                  skipped_duplicate_sources=discovery['skipped_duplicate_sources'],
                  settings=dict(tick_mm=tick_mm,mm_per_pixel=mm_per_pixel,roi=roi,conf=conf,device=device,max_frames=max_frames),
                  smoke_test_only=max_frames is not None)

    def save(current=None,index=0):
        _write_json(output/'batch_report.json', report)
        _write_json(output/'progress.json',dict(state=report['state'],current_video=str(current) if current else None,
                    current_index=index,total_videos=len(sources),completed=report['completed_video_count'],partial=report['partial_video_count'],failed=report['failed_video_count'],
                    current_progress_path=str(output/names[current]/'progress.json') if current else None,
                    output=str(output),fmd_calculated=False))
        write_html(report,output)

    save()
    for index, source in enumerate(sources,1):
        video_output = output/names[source]
        print(f'[{index}/{len(sources)}] 시작: {source.name}',flush=True)
        save(source,index)
        try:
            result = renderer(source=source,weights=checkpoint,output=video_output,tick_mm=tick_mm,
                              mm_per_pixel=mm_per_pixel,roi=roi,conf=conf,device=device,
                              max_frames=max_frames,progress_path=video_output/'progress.json')
            allowed_states = ('complete','partial') if max_frames is not None else ('complete',)
            if not isinstance(result,dict) or result.get('state') not in allowed_states:
                raise RuntimeError('영상 처리기가 완료 상태를 반환하지 않았습니다.')
            json.dumps(result,allow_nan=False)
            result = dict(result,source=str(source),output=str(video_output),fmd_calculated=False)
            report['videos'].append(result)
            report['partial_video_count' if result['state'] == 'partial' else 'completed_video_count'] += 1
            for aggregate, field in [('processed_frame_count','processed'),('measured_frame_count','measured'),('unmeasurable_frame_count','unmeasurable')]:
                report[aggregate] += int(result.get(field,0))
            if video_output.is_dir():
                _write_json(video_output/OUTPUT_MARKER,dict(purpose='diameter_video_output',source=str(source),model=str(checkpoint),fmd_calculated=False))
            state_label = '미리보기 완료' if result['state'] == 'partial' else '완료'
            print(f'[{index}/{len(sources)}] {state_label}: {source.name} · 처리 {result.get("processed",0)} · D {result.get("measured",0)} · 측정 불가 {result.get("unmeasurable",0)}',flush=True)
        except Exception as error:
            failed = dict(state='failed',source=str(source),output=str(video_output),error=f'{type(error).__name__}: {error}',fmd_calculated=False)
            report['videos'].append(failed)
            report['failed_video_count'] += 1
            if video_output.is_dir():
                _write_json(video_output/OUTPUT_MARKER,dict(purpose='diameter_video_output',state='failed',source=str(source),fmd_calculated=False))
            print(f'[{index}/{len(sources)}] 실패: {source.name} · {error}',file=sys.stderr,flush=True)
        save(source,index)
    report['state'] = 'complete_with_errors' if report['failed_video_count'] else ('partial' if report['partial_video_count'] else 'complete')
    report['finished_at'] = datetime.now(KOREA).isoformat(timespec='seconds')
    report['html'] = str(output/'index.html')
    save()
    print(f'결과: {output}',flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description='학습 모델로 원본 영상마다 경계와 직경 D를 표시하고 비교 영상을 저장합니다.')
    result.add_argument('--source',nargs='+',required=True,help='원본 영상 파일 또는 하위 폴더를 포함해 탐색할 폴더')
    result.add_argument('--output',type=Path,help='새 결과 폴더. 기본값: C:\\HJK\\D_분석결과\\한국시간_생성시각')
    result.add_argument('--weights',type=Path,help='사용할 학습 모델. 생략하면 최신 모델 설정을 사용합니다.')
    result.add_argument('--tick-mm',type=float,default=5.0,help='인접 눈금 사이 길이(mm), 기본 5')
    result.add_argument('--mm-per-pixel',type=float,help='직접 지정할 픽셀당 mm 배율')
    result.add_argument('--roi',help='직경 측정 ROI x1,y1,x2,y2. 기본: 초음파 촬영 영역')
    result.add_argument('--conf',type=float,default=.25,help='모델 신뢰도 기준, 기본 0.25')
    result.add_argument('--device',default='0',help='모델 실행 장치, 기본 0')
    result.add_argument('--max-frames',type=int,help=argparse.SUPPRESS)
    return result


def main(argv=None):
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    args = parser().parse_args(argv)
    try:
        roi = [float(value.strip()) for value in args.roi.split(',')] if args.roi else None
        report = run_batch(args.source,args.output,args.weights,args.tick_mm,args.mm_per_pixel,roi,args.conf,args.device,args.max_frames)
    except (ValueError,OSError,ImportError) as error:
        print(f'분석을 시작할 수 없습니다: {error}',file=sys.stderr)
        return 2
    return 1 if report['failed_video_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
