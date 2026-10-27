"""Keep Ultralytics settings and optional integrations local to this project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "work" / "ultralytics_config").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "work" / "ultralytics_config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("COMET_MODE", "DISABLED")
os.environ.setdefault("MLFLOW_TRACKING_URI", "")


def configure():
    from ultralytics import settings
    settings.update({
        "sync": False, "hub": False, "clearml": False, "comet": False,
        "dvc": False, "mlflow": False, "neptune": False, "raytune": False,
        "tensorboard": False, "wandb": False,
        "datasets_dir": str(ROOT / "data" / "export"),
        "weights_dir": str(ROOT / "models"),
        "runs_dir": str(ROOT / "models" / "runs"),
    })


def resolve(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()
