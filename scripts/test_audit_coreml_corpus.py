import unittest

from PIL import Image

from audit_coreml_corpus import prepare_image


class AuditCoreMLCorpusTests(unittest.TestCase):
    def test_yolox_uses_model_input_edge_and_preserves_aspect_ratio(self):
        prepared, content_size = prepare_image(Image.new("RGB", (1000, 500)), "yolox", 320)
        self.assertEqual((prepared.size, content_size), ((320, 320), (320, 160)))


if __name__ == "__main__":
    unittest.main()
