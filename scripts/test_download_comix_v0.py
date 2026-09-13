import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from download_comix_v0 import download_shards


class DownloadComixV0Tests(unittest.TestCase):
    def test_downloads_and_verifies_a_selected_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            data = b"verified shard"
            (source / "shard.tar").write_bytes(data)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "downloadBaseURL": source.as_uri(),
                "shards": [{"file": "shard.tar", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
            }))

            self.assertEqual(download_shards(manifest, output, ["shard.tar"]), ["shard.tar"])
            self.assertEqual((output / "shard.tar").read_bytes(), data)
            self.assertEqual(download_shards(manifest, output, ["shard.tar"]), [])


if __name__ == "__main__":
    unittest.main()
