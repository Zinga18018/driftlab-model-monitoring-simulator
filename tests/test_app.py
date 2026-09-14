"""Smoke-check actual Streamlit controls after changing monitoring semantics."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_default_and_no_drift_scenario_render(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].label, "Held-out reference AUC")
        app.sidebar.selectbox[0].set_value("none").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Separate synthetic windows" in caption.value for caption in app.caption))


if __name__ == "__main__":
    unittest.main()
