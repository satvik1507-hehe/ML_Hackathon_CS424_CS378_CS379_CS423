"""Small synthetic checks for raw-file parsing, candidate labels, and Viterbi decoding."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathsnap.data.reader import load_trip
from pathsnap.decoder.viterbi import decode_viterbi, decoded_metrics
from pathsnap.features.candidates import build_candidate_dataset


class PipelineTests(unittest.TestCase):
    def _write_trip(self, root: Path) -> str:
        trip_id = "00000000"
        folder = root / trip_id
        folder.mkdir(parents=True)
        np.savetxt(folder / f"{trip_id}.track",
                   np.asarray([[0.0001, 0.0, 0.0], [0.0004, 0.0, 1.0],
                               [0.0007, 0.0, 2.0], [0.0009, 0.0, 3.0]]))
        np.savetxt(folder / f"{trip_id}.nodes",
                   np.asarray([[0.0, 0.0], [0.0005, 0.0], [0.001, 0.0]]))
        np.savetxt(folder / f"{trip_id}.arcs", np.asarray([[0, 1], [1, 2]]), fmt="%d")
        np.savetxt(folder / f"{trip_id}.route", np.asarray([0, 1]), fmt="%d")
        return trip_id

    def test_reader_candidate_features_and_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trip_id = self._write_trip(root)
            trip = load_trip(root, trip_id)
            self.assertEqual(trip.gps.shape, (4, 3))
            self.assertEqual(trip.arcs.shape, (2, 2))

            dataset = build_candidate_dataset(trip, top_k=2, search_radius_m=100,
                                              label_radius_m=100)
            self.assertEqual(dataset.X.shape[1], 7)
            self.assertEqual(len(dataset.reference_arc_ids), len(trip.gps))
            self.assertTrue(np.any(dataset.y == 1))
            self.assertLessEqual(float(dataset.candidate_miss.mean()), 0.25)

            scores = np.where(dataset.y == 1, 0.9, 0.1)
            prediction = decode_viterbi(trip, dataset, scores)
            metrics = decoded_metrics(trip, dataset, prediction)
            self.assertEqual(len(prediction), len(trip.gps))
            self.assertGreaterEqual(metrics["top1_accuracy"], 0.75)
            self.assertGreaterEqual(metrics["route_continuity"], 0.5)


if __name__ == "__main__":
    unittest.main()

