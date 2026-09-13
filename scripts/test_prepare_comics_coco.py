import hashlib
import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from prepare_comics_coco import build_dataset


class PrepareComicsCocoTests(unittest.TestCase):
    def test_converts_boxes_and_preserves_fallback_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            image = Image.new("RGB", (100, 80), "white")
            data = BytesIO()
            image.save(data, "JPEG")
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("Images/panel.jpg", data.getvalue())
                output.writestr("Images/fallback.jpg", data.getvalue())
                output.writestr("Annotations/panel.txt", "1 0 0 0 0\n1 10 20 70 60\n")

            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            destination = root / "dataset"
            summary = build_dataset(archive, destination, digest)
            payloads = [json.loads(path.read_text()) for path in (destination / "annotations").glob("*.json")]

            self.assertEqual(summary["panelBoxes"], 1)
            self.assertEqual(summary["fallbackPages"], 1)
            self.assertEqual(sum(len(payload["images"]) for payload in payloads), 2)
            self.assertEqual([item["bbox"] for payload in payloads for item in payload["annotations"]], [[10, 20, 60, 40]])


if __name__ == "__main__":
    unittest.main()
