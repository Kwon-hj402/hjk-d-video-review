"""Local-only FMD annotation workbench. No image is sent to an external service."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import importlib.util
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
# Trusted local project paths allow helper modules to resolve their own imports
# whether started as app/server.py or imported by local tests and the MCP bridge.
for project_path in (ROOT, ROOT / "scripts"):
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
METADATA = DATA / "video_metadata.json"
LOCK = threading.RLock()
MANIFEST_CACHE = {"mtime": None, "value": None}
AUTO_CACHE = {"key": None, "value": None, "by_id": {}}
ASSIST_MODULE = None
app = FastAPI(title="FMD 영상 레이블링 작업대")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    messages = []
    for error in exc.errors():
        message = error.get("msg", "")
        if message.startswith("Value error, "):
            messages.append(message[len("Value error, "):])
        else:
            names = {"mm_per_pixel": "길이 보정값", "cuff_release_video_seconds": "커프 해제 시각", "cuff_inflation_video_seconds": "커프 가압 시각", "points": "윤곽 좌표", "box": "AI 영역 좌표", "frame_id": "프레임 ID"}
            location = str(error.get("loc", [""])[-1])
            messages.append(f"{names.get(location, '입력값')} 형식을 확인하세요.")
    return JSONResponse({"detail": " ".join(dict.fromkeys(messages))}, status_code=422)


@app.middleware("http")
async def local_only(request: Request, call_next):
    # Block cross-site requests to a loopback service, including DNS-rebinding hosts.
    host = request.headers.get("host", "").split(":")[0].lower()
    if host not in {"127.0.0.1", "localhost", "testserver"}:
        return JSONResponse({"detail": "로컬 주소에서만 사용할 수 있습니다."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and (urlparse(origin).hostname not in {"127.0.0.1", "localhost"} or urlparse(origin).port != request.url.port):
        return JSONResponse({"detail": "외부 웹사이트의 요청을 허용하지 않습니다."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"저장 파일을 읽을 수 없습니다: {path.name}") from exc


def atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = DATA / "annotation_backups" / path.stem
        backup.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (backup / f"{stamp}.json").write_bytes(path.read_bytes())
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def manifest():
    path = DATA / "manifest.json"
    stamp = path.stat().st_mtime_ns if path.exists() else None
    if MANIFEST_CACHE["value"] is not None and MANIFEST_CACHE["mtime"] == stamp:
        return MANIFEST_CACHE["value"]
    value = read_json(path, {"version": 1, "frames": [], "videos": []})
    if not isinstance(value.get("frames"), list):
        raise HTTPException(500, "프레임 목록 형식이 올바르지 않습니다.")
    MANIFEST_CACHE.update(mtime=stamp, value=value)
    return value


def frame_by_id(frame_id: str):
    if not frame_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in frame_id):
        raise HTTPException(400, "프레임 ID가 올바르지 않습니다.")
    for frame in manifest()["frames"]:
        if frame.get("id") == frame_id:
            return frame
    raise HTTPException(404, "프레임을 찾을 수 없습니다. 추출 완료 여부를 확인해 주세요.")


def asset_path(frame: dict, kind: str):
    relative = frame.get(kind)
    if not relative:
        raise HTTPException(404, "영상 파일 경로가 없습니다.")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(DATA.resolve()) or not path.is_file():
        raise HTTPException(404, "프레임 이미지가 아직 준비되지 않았습니다.")
    return path


class Meta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    patient_id: str = "unassigned"
    session_id: str = "unassigned"
    phase: str = "unknown"
    anatomy: str = "unconfirmed"
    mm_per_pixel: float | None = None
    cuff_release_video_seconds: float | None = None
    cuff_inflation_video_seconds: float | None = None

    @field_validator("patient_id", "session_id")
    @classmethod
    def clean_identifier(cls, value):
        value = value.strip()
        if len(value) > 100 or any(ord(c) < 32 for c in value):
            raise ValueError("식별자는 100자 이내로 입력하세요.")
        return value or "unassigned"

    @field_validator("phase")
    @classmethod
    def valid_phase(cls, value):
        if value not in {"unknown", "baseline", "post_release"}:
            raise ValueError("촬영 단계가 올바르지 않습니다.")
        return value

    @field_validator("anatomy")
    @classmethod
    def valid_anatomy(cls, value):
        if value not in {"unconfirmed", "brachial", "carotid", "other"}:
            raise ValueError("혈관 종류가 올바르지 않습니다.")
        return value

    @field_validator("mm_per_pixel")
    @classmethod
    def valid_scale(cls, value):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("배율은 0보다 큰 mm/px 값이어야 합니다.")
        return value

    @field_validator("cuff_release_video_seconds", "cuff_inflation_video_seconds")
    @classmethod
    def valid_release(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("커프 시각은 영상 시작 기준 초 단위로 입력하세요.")
        return value


class Polygon(BaseModel):
    class_id: int = 0
    label: str = "lumen"
    points: list[list[float]]


class Annotation(Meta):
    version: int = 1
    frame_id: str
    status: str = "draft"
    notes: str = Field(default="", max_length=4000)
    objects: list[Polygon] = Field(default_factory=list, max_length=30)
    negative: bool = False


def generated_index():
    path = DATA / "auto_generated" / "index.json"
    key = (str(path), path.stat().st_mtime_ns if path.exists() else None)
    if AUTO_CACHE["key"] == key:
        return AUTO_CACHE["value"], AUTO_CACHE["by_id"]
    value = read_json(path, {"frames": [], "complete": False, "generated_count": 0, "failed_count": 0})
    by_id = {f.get("frame_id", f.get("id")): f for f in value.get("frames", [])}
    AUTO_CACHE.update(key=key, value=value, by_id=by_id)
    return value, by_id


def generated_proposal(frame_id: str):
    # Resolve by known frame id; index paths cannot redirect this read elsewhere.
    value = read_json(DATA / "auto_generated" / "proposals" / f"{frame_id}.json", None)
    if not value or not value.get("objects"):
        return None
    return value


def effective_annotation(frame: dict, metadata=None, saved=None, include_auto=False):
    stored = read_json(ANNOTATIONS / f"{frame['id']}.json", None) if saved is None else saved.get(frame["id"])
    defaults = Meta(patient_id=frame.get("patient_id", "unassigned")).model_dump()
    metadata = read_json(METADATA, {}) if metadata is None else metadata
    metadata = metadata.get("videos", metadata)
    video_meta = metadata.get(frame.get("video_id"), {})
    defaults.update({k: v for k, v in video_meta.items() if k in defaults})
    result = {"version": 1, "frame_id": frame["id"], "status": "unlabeled", "origin": "none", "notes": "", "objects": [], "negative": False, **defaults}
    if stored is not None:
        result.update(stored)
        result["origin"] = "user"
        # Video metadata intentionally acts as shared session metadata. Existing
        # frame-specific values are retained unless the video metadata was applied later.
        if video_meta.get("updated_at", "") > stored.get("updated_at", ""):
            result.update({k: v for k, v in video_meta.items() if k in defaults})
    elif include_auto:
        proposal = generated_proposal(frame["id"])
        if proposal:
            result.update({k: proposal[k] for k in ("objects", "diagnostics", "warnings", "questionable", "model", "prompt") if k in proposal})
            result.update(status="draft", origin="auto", review_required=True, approved=False)
    result["video_metadata"] = video_meta
    return result


@app.get("/")
def home():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    doc = manifest()
    _, auto = generated_index()
    return {"ok": True, "frame_count": len(doc["frames"]), "selected_count": sum(bool(f.get("selected")) for f in doc["frames"]), "annotation_count": len(list(ANNOTATIONS.glob("*.json"))), "auto_generated_count": len(auto), "manifest_ready": bool(doc["frames"]), "local_only": True, "root": str(ROOT)}


@app.get("/api/frames")
def frames(selected: bool = True, mode: str | None = None):
    mode = mode or ("selected" if selected else "all")
    if mode not in {"selected", "all", "generated"}:
        raise HTTPException(400, "프레임 보기 방식은 selected, all, generated 중 선택하세요.")
    doc = manifest()
    rows = []
    _, auto = generated_index()
    totals = {"all": len(doc["frames"]), "selected": 0, "annotated": 0, "approved": 0, "excluded": 0, "patients": 0, "auto_generated": len(auto), "auto_questionable": sum(bool(v.get("questionable")) for v in auto.values())}
    patients = set()
    metadata = read_json(METADATA, {})
    saved = {p.stem: read_json(p, {}) for p in ANNOTATIONS.glob("*.json")}
    for frame in doc["frames"]:
        # Generated drafts are separate from the user's canonical annotations;
        # list summaries need no read of thousands of proposal polygon files.
        annotation = effective_annotation(frame, metadata, saved, include_auto=False)
        status = annotation["status"]
        totals["selected"] += int(bool(frame.get("selected")))
        totals["annotated"] += int(status in {"draft", "approved"})
        totals["approved"] += int(status == "approved")
        totals["excluded"] += int(status == "excluded")
        if status == "approved" and annotation["patient_id"] != "unassigned":
            patients.add(annotation["patient_id"])
        generated = auto.get(frame["id"])
        if mode == "selected" and not frame.get("selected") or mode == "generated" and not generated:
            continue
        origin = annotation["origin"]
        if mode == "generated" and generated and frame["id"] not in saved:
            status, origin = "draft", "auto"
        rows.append({**frame, "status": status, "origin": origin, "auto_generated": bool(generated), "questionable": bool(generated and generated.get("questionable")), "auto_warnings": generated.get("warnings", []) if generated else [], "object_count": len(annotation["objects"]) if origin != "auto" else generated.get("object_count", 1), "patient_id": annotation["patient_id"], "phase": annotation["phase"], "anatomy": annotation["anatomy"]})
    totals["patients"] = len(patients)
    rows.sort(key=lambda f: (f.get("video_id", ""), f.get("frame_index", 0)))
    return {"frames": rows, "videos": doc.get("videos", []), "totals": totals, "mode": mode}


@app.get("/api/batch-status")
def batch_status():
    value = read_json(DATA / "auto_generated" / "status.json", {})
    processed = value.get("processed", value.get("attempted", 0))
    total = value.get("total", len(manifest()["frames"]))
    running = value.get("running", value.get("state") == "running")
    state = value.get("state", "running" if running else "not_started")
    return {**value, "total": total, "processed": processed, "generated": value.get("generated", 0), "failed": value.get("failed", 0), "state": state, "running": running, "review_required": True}


@app.get("/api/image/{frame_id}")
def original(frame_id: str):
    return FileResponse(asset_path(frame_by_id(frame_id), "image_path"))


@app.get("/api/thumbnail/{frame_id}")
def thumbnail(frame_id: str, generated: bool = False):
    frame = frame_by_id(frame_id)
    if generated:
        _, auto = generated_index()
        candidate = auto.get(frame_id, {})
        if candidate.get("thumbnail_path"):
            try:
                return FileResponse(asset_path(candidate, "thumbnail_path"))
            except HTTPException:
                pass
    return FileResponse(asset_path(frame, "thumbnail_path" if frame.get("thumbnail_path") else "image_path"))


@app.get("/api/annotation/{frame_id}")
def annotation(frame_id: str, include_auto: bool = False):
    return effective_annotation(frame_by_id(frame_id), include_auto=include_auto)


@app.post("/api/annotation/{frame_id}")
def save_annotation(frame_id: str, value: Annotation):
    frame = frame_by_id(frame_id)
    if value.frame_id != frame_id or value.status not in {"draft", "approved", "excluded"}:
        raise HTTPException(400, "레이블 ID 또는 상태가 올바르지 않습니다.")
    if value.status == "approved" and not value.objects and not value.negative:
        raise HTTPException(400, "승인하려면 내강 경계를 그리거나 ‘내강 없음’을 명시하세요.")
    if value.negative and value.objects:
        raise HTTPException(400, "내강 없음과 내강 윤곽을 동시에 저장할 수 없습니다.")
    width, height = frame["width"], frame["height"]
    for obj in value.objects:
        if obj.class_id != 0 or len(obj.points) < 3 or len(obj.points) > 5000:
            raise HTTPException(400, "내강 윤곽은 최소 3개 꼭짓점으로 그려 주세요.")
        for point in obj.points:
            if len(point) != 2 or not all(math.isfinite(x) for x in point) or not (0 <= point[0] <= width - 1 and 0 <= point[1] <= height - 1):
                raise HTTPException(400, "윤곽 좌표가 원본 이미지 범위를 벗어났습니다.")
        area = abs(sum(obj.points[i][0] * obj.points[(i + 1) % len(obj.points)][1] - obj.points[(i + 1) % len(obj.points)][0] * obj.points[i][1] for i in range(len(obj.points)))) / 2
        if area < 1:
            raise HTTPException(400, "면적이 없는 윤곽은 저장할 수 없습니다.")
    result = value.model_dump()
    result["origin"] = "user"
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    with LOCK:
        atomic_json(ANNOTATIONS / f"{frame_id}.json", result)
    return {"ok": True, "annotation": result}


@app.get("/api/video-metadata/{video_id}")
def get_video_metadata(video_id: str):
    if not any(f.get("video_id") == video_id for f in manifest()["frames"]):
        raise HTTPException(404, "영상을 찾을 수 없습니다.")
    return read_json(METADATA, {}).get(video_id, Meta().model_dump())


@app.post("/api/video-metadata/{video_id}")
def save_video_metadata(video_id: str, value: Meta):
    get_video_metadata(video_id)
    with LOCK:
        existing = read_json(METADATA, {})
        existing[video_id] = {**value.model_dump(), "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(METADATA, existing)
    return {"ok": True, "metadata": existing[video_id]}


class ExportOptions(BaseModel):
    exploratory: bool = False
    mode: str | None = None
    labels_only: bool = False


@app.post("/api/export")
def export_dataset(options: ExportOptions):
    script = ROOT / "scripts" / "export_dataset.py"
    if not script.exists():
        raise HTTPException(503, "데이터셋 내보내기 도구를 준비하고 있습니다.")
    mode = options.mode or ("video-exploratory" if options.exploratory else "patient")
    if mode not in {"patient", "video-exploratory"}:
        raise HTTPException(400, "분할 방식은 patient 또는 video-exploratory를 선택하세요.")
    command = [sys.executable, str(script), "--root", str(ROOT), "--mode", mode]
    if options.labels_only:
        command.append("--labels-only")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise HTTPException(400, output[-3000:] or "내보내기를 완료하지 못했습니다.")
    return {"ok": True, "message": output[-3000:], "path": str(DATA / "export" / "current")}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.get("/api/measurement/{frame_id}")
def measure_frame(frame_id: str):
    frame = frame_by_id(frame_id)
    ann = effective_annotation(frame)
    module = load_module("fmd_measurement", ROOT / "scripts" / "measurement.py")
    result = module.measure_annotation(ann, frame)
    result["available"] = result.get("mean_diameter_px") is not None
    return result


@app.get("/api/fmd-summary")
def fmd_summary():
    metadata = read_json(METADATA, {})
    saved = {p.stem: read_json(p, {}) for p in ANNOTATIONS.glob("*.json")}
    rows = [(f, effective_annotation(f, metadata, saved)) for f in manifest()["frames"] if f["id"] in saved]
    module = load_module("fmd_measurement", ROOT / "scripts" / "measurement.py")
    return module.summarize_fmd(rows)


class AssistBox(BaseModel):
    box: list[float] = Field(min_length=4, max_length=4)


@app.post("/api/assist/{frame_id}")
def propose_boundary(frame_id: str, options: AssistBox):
    global ASSIST_MODULE
    frame = frame_by_id(frame_id)
    x1, y1, x2, y2 = options.box
    if not all(math.isfinite(x) for x in options.box) or not (0 <= x1 < x2 <= frame["width"] - 1 and 0 <= y1 < y2 <= frame["height"] - 1):
        raise HTTPException(400, "원본 이미지 안에서 내강 영역을 지정해 주세요.")
    script = ROOT / "scripts" / "assist.py"
    if not script.exists():
        raise HTTPException(503, "AI 경계 모델을 준비하고 있습니다. 수동 윤곽을 먼저 그릴 수 있습니다.")
    try:
        # Model creation and inference share a lock to avoid simultaneous CPU/GPU allocation.
        with LOCK:
            if ASSIST_MODULE is None:
                ASSIST_MODULE = load_module("fmd_assist", script)
            result = ASSIST_MODULE.propose(str(asset_path(frame, "image_path")), box=options.box)
        return {**result, "status": "draft", "review_required": True}
    except Exception as exc:
        raise HTTPException(503, f"AI 경계 제안을 실행하지 못했습니다: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
