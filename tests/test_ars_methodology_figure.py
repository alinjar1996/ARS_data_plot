"""Lightweight checks for the editable, text-only methodology figure."""

import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt

from make_ars_methodology_figure import (
    BLOCKS, LABELS, STEM, build_figure, save_figure, validate_labels,
)


class MethodologyFigureTests(unittest.TestCase):
    def setUp(self):
        self.fig = build_figure()
        self.ax = self.fig.axes[0]

    def tearDown(self):
        plt.close(self.fig)

    def test_single_opaque_planner_and_no_equations(self):
        self.assertEqual(set(BLOCKS), {"policy", "planner", "environment", "training"})
        self.assertEqual(LABELS["planner"], "MPPI")
        self.assertEqual(len(self.ax.images), 0)
        for label in LABELS.values():
            self.assertNotIn("$", label)
            self.assertNotIn("=", label)

    def test_label_spacing_and_block_padding(self):
        validate_labels(self.fig)

    def test_online_and_training_connections(self):
        arrows = {patch.get_gid(): patch for patch in self.ax.patches
                  if (patch.get_gid() or "").startswith("arrow-")}
        solid = {"cost-weights", "control-action", "state-to-policy", "state-to-planner"}
        dashed = {"policy-parameters", "episode-returns"}
        self.assertEqual(set(arrows), {"arrow-" + key for key in solid | dashed})
        for key in solid:
            self.assertEqual(arrows["arrow-" + key].get_linestyle(), "-")
        for key in dashed:
            self.assertNotEqual(arrows["arrow-" + key].get_linestyle(), "-")

    def test_exports_keep_editable_text_and_protect_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = save_figure(directory, dpi=60)
            self.assertEqual({p.suffix for p in paths}, {".svg", ".pdf", ".png"})
            self.assertTrue(all(p.stat().st_size > 1000 for p in paths))
            svg = ElementTree.parse(Path(directory) / f"{STEM}.svg")
            texts = ["".join(node.itertext()) for node in
                     svg.iter("{http://www.w3.org/2000/svg}text")]
            self.assertIn("MPPI", texts)
            self.assertIn("ARS training", texts)
            self.assertIn("Cost weights", texts)
            before = [p.read_bytes() for p in paths]
            with self.assertRaises(FileExistsError):
                save_figure(directory, dpi=60)
            self.assertEqual(before, [p.read_bytes() for p in paths])


if __name__ == "__main__":
    unittest.main()
