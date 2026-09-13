import unittest

from prepare_private_panel_audit import ordered_images, rectangle


class PrivatePanelAuditTests(unittest.TestCase):
    def test_ordering_and_normalized_rectangle_contract(self):
        self.assertEqual(
            ordered_images(["page10.jpg", "__MACOSX/page1.jpg", "page2.jpg", ".hidden/page3.png"]),
            ["page2.jpg", "page10.jpg"],
        )
        self.assertEqual(rectangle([[0.125, 0.25], [0.25, 0.5]]), [0.125, 0.25, 0.375, 0.75])


if __name__ == "__main__":
    unittest.main()
