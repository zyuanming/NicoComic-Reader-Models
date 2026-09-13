import unittest

from scripts.score_panel_truth import maximum_match, ordered, postprocess


class PanelTruthScorerTests(unittest.TestCase):
    def test_matches_one_to_one_and_rejects_unsafe_detections(self):
        expected = [[0, 0, 0.5, 1], [0.5, 0, 1, 1]]
        indices, values = maximum_match(expected, list(reversed(expected)))
        self.assertEqual(indices, [1, 0])
        self.assertEqual(values, [1, 1])
        self.assertEqual(maximum_match(expected, [expected[1]])[0], [None, 0])
        self.assertEqual(
            ordered(
                [[0.4, -0.01, 0.9, 0.19], [0, 0, 0.4, 0.36]],
                "rightToLeft",
            )[0],
            [0, 0, 0.4, 0.36],
        )

        self.assertEqual(
            postprocess(
                [
                    {"score": 0.95, "box": [0, 0, 0.7, 1]},
                    {"score": 0.95, "box": [0.3, 0, 1, 1]},
                ]
            ),
            [],
        )
        self.assertEqual(
            postprocess(
                [
                    {"score": 0.79, "box": [0, 0, 0.4, 0.4]},
                    {"score": 0.95, "box": [0, 0.5, 0.4, 0.55]},
                ]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
