import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_comix_v0_coco import build_dataset


class PrepareComixV0CocoTests(unittest.TestCase):
    def test_verifies_shard_and_converts_story_and_fallback_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "shard.tar"
            with tarfile.open(archive, "w") as target:
                for page_id, page_class, panels in [
                    ("book-a-p001", "story", [[10, 20, 90, 70], [0, 0, 8, 10]]),
                    ("book-b-p001", "cover", [[0, 0, 100, 80]]),
                    ("book-c-p001", "first-page", [[0, 0, 100, 80]]),
                ]:
                    image = Image.new("RGB", (100, 80), "white")
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, "JPEG")
                    metadata = json.dumps({
                        "page_id": page_id,
                        "book_id": page_id.split("-p")[0],
                        "page_class": page_class,
                        "image": {"file": f"{page_id}.jpg", "width": 100, "height": 80},
                        "detections": {"fasterrcnn": {"panels": [{"bbox": box} for box in panels]}},
                    }).encode()
                    for name, data in [(f"{page_id}.json", metadata), (f"{page_id}.jpg", image_bytes.getvalue())]:
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        target.addfile(member, io.BytesIO(data))

            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "sourceURL": "https://example.test/dataset",
                "repositoryRevision": "fixed-revision",
                "license": "CC0-1.0",
                "annotationKind": "pseudo-label",
                "shards": [{"file": archive.name, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}],
            }))
            output = root / "output"

            summary = build_dataset(manifest, [archive], output)
            payloads = [json.loads(path.read_text()) for path in (output / "annotations").glob("*.json")]

            self.assertEqual(summary["pagesSeen"], 3)
            self.assertEqual(summary["pagesIncluded"], 2)
            self.assertEqual(summary["positivePages"], 1)
            self.assertEqual(summary["negativePages"], 1)
            self.assertEqual(summary["panelBoxes"], 2)
            self.assertEqual(summary["excludedClasses"], {"first-page": 1})
            self.assertEqual(
                [item["bbox"] for payload in payloads for item in payload["annotations"]],
                [[10.0, 20.0, 80.0, 50.0], [0.0, 0.0, 8.0, 10.0]],
            )


if __name__ == "__main__":
    unittest.main()
