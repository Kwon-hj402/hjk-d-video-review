"""Storage and coordinate regressions, entirely isolated from real annotations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

SERVER_PATH = Path(__file__).with_name("server.py")
spec = importlib.util.spec_from_file_location("workbench_server", SERVER_PATH)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


class AnnotationStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.frame = {"id": "v01_f000010", "video_id": "v01", "width": 800, "height": 600, "selected": True, "timestamp_seconds": 1, "image_path": "data/test.png"}
        (self.data / "manifest.json").write_text(json.dumps({"frames": [self.frame], "videos": []}), encoding="utf-8")
        self.patches = [patch.object(server, "ROOT", self.root), patch.object(server, "DATA", self.data), patch.object(server, "ANNOTATIONS", self.data / "annotations"), patch.object(server, "METADATA", self.data / "video_metadata.json")]
        for p in self.patches:
            p.start()
        server.MANIFEST_CACHE.update(mtime=None, value=None)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        server.MANIFEST_CACHE.update(mtime=None, value=None)
        self.temp.cleanup()

    def value(self, **updates):
        payload = {"frame_id": self.frame["id"], "objects": [{"class_id": 0, "points": [[110.25, 130.5], [680.75, 130.5], [680.75, 190.5], [110.25, 190.5]]}]}
        payload.update(updates)
        return server.Annotation(**payload)

    def test_original_coordinates_preserved_and_backup(self):
        server.save_annotation(self.frame["id"], self.value(status="draft"))
        server.save_annotation(self.frame["id"], self.value(status="approved"))
        stored = server.annotation(self.frame["id"])
        self.assertEqual(stored["objects"][0]["points"][0], [110.25, 130.5])
        self.assertEqual(stored["status"], "approved")
        self.assertEqual(len(list((self.data / "annotation_backups" / self.frame["id"]).glob("*.json"))), 1)

    def test_empty_approved_is_blocked_and_negative_is_allowed(self):
        with self.assertRaises(server.HTTPException):
            server.save_annotation(self.frame["id"], self.value(status="approved", objects=[]))
        server.save_annotation(self.frame["id"], self.value(status="approved", objects=[], negative=True))
        self.assertTrue(server.annotation(self.frame["id"])["negative"])

    def test_negative_cannot_contain_lumen(self):
        with self.assertRaises(server.HTTPException):
            server.save_annotation(self.frame["id"], self.value(negative=True))

    def test_invalid_original_coordinate_is_blocked(self):
        with self.assertRaises(server.HTTPException):
            server.save_annotation(self.frame["id"], self.value(objects=[{"points": [[0, 0], [800, 1], [1, 20]]}]))

    def test_later_video_metadata_is_effective_and_signed_times(self):
        server.save_annotation(self.frame["id"], self.value(patient_id="P01"))
        server.save_video_metadata("v01", server.Meta(patient_id="P02", cuff_release_video_seconds=-60, cuff_inflation_video_seconds=-360))
        effective = server.annotation(self.frame["id"])
        self.assertEqual(effective["patient_id"], "P02")
        self.assertEqual(effective["cuff_release_video_seconds"], -60)
        totals = server.frames()["totals"]
        self.assertEqual(totals["selected"], 1)
        self.assertEqual(totals["annotated"], 1)

    def test_image_path_cannot_escape_data(self):
        (self.root / "secret.txt").write_text("private", encoding="utf-8")
        with self.assertRaises(server.HTTPException):
            server.asset_path({"image_path": "secret.txt"}, "image_path")
        with self.assertRaises(server.HTTPException):
            server.frame_by_id("../secret")

    def test_auto_drafts_are_visible_without_creating_human_annotation(self):
        auto = self.data / "auto_generated"
        (auto / "proposals").mkdir(parents=True)
        (auto / "index.json").write_text(json.dumps({"frames": [{"frame_id": self.frame["id"], "questionable": True}]}), encoding="utf-8")
        proposal = {"status": "approved", "objects": self.value().model_dump()["objects"], "review_required": True}
        (auto / "proposals" / f"{self.frame['id']}.json").write_text(json.dumps(proposal), encoding="utf-8")
        manual = server.annotation(self.frame["id"])
        self.assertEqual(manual["objects"], [])
        self.assertEqual(manual["status"], "unlabeled")
        effective = server.annotation(self.frame["id"], include_auto=True)
        self.assertEqual(effective["origin"], "auto")
        self.assertEqual(effective["status"], "draft")
        self.assertFalse((self.data / "annotations" / f"{self.frame['id']}.json").exists())
        gallery = server.frames(mode="generated")
        self.assertEqual(len(gallery["frames"]), 1)
        self.assertEqual(gallery["totals"]["auto_generated"], 1)
        self.assertEqual(gallery["totals"]["annotated"], 0)
        self.assertEqual(gallery["totals"]["approved"], 0)
        self.assertEqual(gallery["frames"][0]["origin"], "auto")
        self.assertTrue(gallery["frames"][0]["questionable"])
        server.save_annotation(self.frame["id"], self.value(status="excluded", objects=[]))
        canonical = server.annotation(self.frame["id"])
        self.assertEqual(canonical["origin"], "user")
        self.assertEqual(canonical["status"], "excluded")
        self.assertEqual(canonical["objects"], [])
        self.assertEqual(server.frames(mode="generated")["frames"][0]["status"], "excluded")

    def test_batch_progress_aliases(self):
        auto = self.data / "auto_generated"
        auto.mkdir()
        (auto / "status.json").write_text(json.dumps({"total": 20, "attempted": 10, "generated": 8, "failed": 2, "running": True}), encoding="utf-8")
        status = server.batch_status()
        self.assertEqual(status["processed"], 10)
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["generated"], 8)


if __name__ == "__main__":
    unittest.main()
