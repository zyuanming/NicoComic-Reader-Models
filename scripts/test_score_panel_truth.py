import unittest

from scripts.score_panel_truth import maximum_match, ordered, postprocess, score


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
            [0.4, 0.0, 0.9, 0.19],
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

    def test_scores_panel_and_fallback_pages_separately(self):
        truth = {
            "readingDirection": "rightToLeft",
            "pages": [
                {
                    "id": "panel",
                    "file": "panel.jpg",
                    "expectation": "panels",
                    "regions": [
                        {"order": 0, "box": [0.6, 0, 1, 1]},
                        {"order": 1, "box": [0, 0, 0.4, 1]},
                    ],
                },
                {
                    "id": "fallback",
                    "file": "fallback.jpg",
                    "expectation": "fallback",
                    "regions": [],
                },
            ],
        }
        audit = {
            "pages": [
                {
                    "page": "panel.jpg",
                    "detections": [
                        {"score": 0.9, "box": [0.6, 0, 1, 1]},
                        {"score": 0.9, "box": [0, 0, 0.4, 1]},
                    ],
                },
                {"page": "fallback.jpg", "detections": []},
            ]
        }
        result = score(truth, audit, minimum_score=0.85)
        self.assertEqual(
            (result["panelNoModificationRate"], result["fallbackAccuracy"], result["minimumScore"]),
            (1.0, 1.0, 0.85),
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
