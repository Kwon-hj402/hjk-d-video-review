"""Verify local dependencies and actual GPU training. Synthetic images only."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone

try:
    from ._runtime import ROOT, configure
except ImportError:
    from _runtime import ROOT, configure


def synthetic_training():
    import cv2
    import numpy as np
    import yaml
    from ultralytics import YOLO
    fixture = ROOT / "work" / "synthetic_smoke"
    rng = np.random.default_rng(42)
    for split, count in (("train", 6), ("val", 2)):
        (fixture / "images" / split).mkdir(parents=True, exist_ok=True)
        (fixture / "labels" / split).mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            im = rng.integers(30, 100, size=(128, 128, 3), dtype=np.uint8)
            cx, cy = 64 + idx % 3 - 1, 64 + idx % 2
            rx, ry = 39, 16 + idx % 3
            angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            polygon = np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])
            cv2.fillPoly(im, [polygon.astype(np.int32)], (6, 6, 6))
            cv2.polylines(im, [polygon.astype(np.int32)], True, (180, 180, 180), 2)
            cv2.imencode(".png", im)[1].tofile(fixture / "images" / split / f"synthetic_{idx}.png")
            line = "0 " + " ".join(f"{float(x):.6f}" for x in (polygon / 128).ravel())
            (fixture / "labels" / split / f"synthetic_{idx}.txt").write_text(line + "\n", encoding="utf-8")
    dataset = fixture / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"path": str(fixture), "train": "images/train", "val": "images/val", "names": {0: "lumen"}}), encoding="utf-8")
    (fixture / "SYNTHETIC_ONLY.txt").write_text("Synthetic environment fixture. No patient data. Not an ultrasound/FMD model.\n", encoding="utf-8")
    model = YOLO(str(ROOT / "models" / "yolo11s-seg.pt"))
    config = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
    config.pop("model")
    config.update(data=str(dataset), epochs=1, imgsz=128, batch=2, device=0, workers=0,
                  project=str(fixture / "runs"), name="pipeline", exist_ok=True, plots=False,
                  save_period=-1, patience=1, val=True, verbose=False)
    start = time.perf_counter()
    model.train(**config)
    run_dir = model.trainer.save_dir
    metadata = {"purpose": "synthetic_pipeline_smoke", "medical_validation": False, "patient_data_used": False}
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best = run_dir / "weights" / "best.pt"
    return {"passed": best.is_file(), "duration_seconds": round(time.perf_counter() - start, 2), "fixture_path": str(fixture), "best_checkpoint": str(best), "training_batches": 3, "patient_data_used": False, "validation_is_medical_performance": False}


def main():
    parser = argparse.ArgumentParser(description="학습 환경 확인 (옵션: 합성 데이터 1 epoch 학습)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure()
    import torch
    packages = ["torch", "torchvision", "ultralytics", "fastapi", "uvicorn", "opencv-python", "numpy", "Pillow", "PyYAML", "mcp"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(), "versions": versions,
        "isolation": "venv --system-site-packages; GPU PyTorch reused read-only from existing Miniconda base",
        "base_prefix": sys.base_prefix, "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "medical_model_trained": False, "patient_labels_used": False,
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        # An explicit backward + optimizer step checks more than CUDA enumeration.
        layer = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
        before = layer.weight.detach().clone()
        loss = (layer(torch.randn(2, 1, 64, 64, device="cuda")) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step(); torch.cuda.synchronize()
        report["gpu_backward_optimizer_test"] = {"passed": bool(not torch.equal(before, layer.weight)), "loss": float(loss.detach())}
    for name in ("yolo11s-seg.pt", "sam2.1_t.pt"):
        path = ROOT / "models" / name
        report.setdefault("weights", {})[name] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
    if args.smoke:
        if not report["cuda_available"]:
            raise RuntimeError("Synthetic GPU training smoke requires CUDA.")
        report["segmentation_training_smoke"] = synthetic_training()
        (ROOT / "reports" / "environment_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out = ROOT / "reports" / "environment.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
