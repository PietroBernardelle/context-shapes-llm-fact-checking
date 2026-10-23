from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/scripts"))

from audit_results import check_coverage, expected_files
from evaluation_utils import filter_configured_metrics, filter_subset, macro_f1_from_counts


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowTests(unittest.TestCase):
    def test_class_names_and_macro_f1(self) -> None:
        tables = _load_script("10_generate_latex_table")
        frame = pd.DataFrame({"label": [0, 0, 0, 1], "pred_label": [0, 0, 1, 1]})
        metrics = tables.get_metrics_for_df(frame)
        self.assertEqual(metrics["Acc"], 0.75)
        self.assertEqual(metrics["Prec_SUP"], 1.0)
        self.assertAlmostEqual(metrics["Rec_SUP"], 2 / 3)
        self.assertEqual(metrics["Prec_REF"], 0.5)
        self.assertEqual(metrics["Rec_REF"], 1.0)
        self.assertAlmostEqual(metrics["F1"], (0.8 + 2 / 3) / 2)
        counts = {"valid": 4, "correct": 3, "tp": 1, "fp": 1, "fn": 0}
        self.assertAlmostEqual(macro_f1_from_counts(counts), metrics["F1"])

    def test_invalid_predictions_use_consistent_denominator(self) -> None:
        tables = _load_script("10_generate_latex_table")
        frame = pd.DataFrame({"label": [0, 1, 1], "pred_label": [0, 1, None]})
        self.assertEqual(tables.get_metrics_for_df(frame)["Acc"], 1.0)

    def test_subset_identity_includes_dataset(self) -> None:
        frame = pd.DataFrame({"dataset": ["Hover", "Feverous"], "claim": ["synthetic", "synthetic"]})
        selected = filter_subset(frame, {("hover", "synthetic")})
        self.assertEqual(selected["dataset"].tolist(), ["Hover"])

    def test_coverage_detects_partial_and_duplicate_rows(self) -> None:
        subset = pd.DataFrame({"dataset": ["Hover"] * 3,
                               "claim": ["synthetic-a", "synthetic-a", "synthetic-b"],
                               "label": [0, 0, 1]})
        complete = subset.assign(pred_label=[0, 0, 1])
        self.assertTrue(check_coverage(complete, subset).complete)
        partial = complete.iloc[[0, 2]]
        self.assertEqual(check_coverage(partial, subset).missing_rows, 1)
        duplicate = complete.iloc[[0, 0, 1]]
        coverage = check_coverage(duplicate, subset)
        self.assertEqual((coverage.missing_rows, coverage.extra_rows), (1, 1))
        incorrect_labels = complete.copy()
        incorrect_labels.loc[0, "label"] = 1
        self.assertFalse(check_coverage(incorrect_labels, subset).complete)
        extra = complete.iloc[:1].assign(claim="synthetic-outside-subset")
        with_extra = pd.concat([complete, extra], ignore_index=True)
        self.assertEqual(check_coverage(with_extra, subset).extra_rows, 1)
        self.assertTrue(check_coverage(with_extra, subset, allow_outside_subset=True).complete)

    def test_paper_grid_excludes_extra_depths_and_contexts(self) -> None:
        model = "Qwen3-8B"
        frame = pd.DataFrame({"model": [model] * 4,
                              "filler_length": [2048, 2048, 32768, 4096],
                              "evidence_position": [0.1, 0.05, 0.1, 0.9]})
        config = ROOT / "config/paper_config.json"
        self.assertEqual(filter_configured_metrics(frame, config).index.tolist(), [0, 3])
        self.assertEqual(filter_configured_metrics(frame, config, True).index.tolist(), [0, 1, 3])
        self.assertTrue(filter_configured_metrics(pd.DataFrame(), config).empty)
        baseline = pd.DataFrame({"model": [model], "filler_length": [0], "evidence_position": [None]})
        self.assertTrue(filter_configured_metrics(baseline, config).empty)

    def test_expected_condition_coverage(self) -> None:
        config = json.loads((ROOT / "config/paper_config.json").read_text())
        paths = expected_files(config, Path("results"))
        self.assertEqual(len(paths), 5 * (44 + 2))
        self.assertEqual(len(set(paths)), len(paths))

    def test_balanced_sampling_fails_on_short_input(self) -> None:
        subset_script = _load_script("6_build_experiment_subset")
        frame = pd.DataFrame({"label": [0, 0, 1], "claim": ["a", "b", "c"]})
        with self.assertRaises(ValueError):
            subset_script.sample_balanced(frame, 2, 42)

    def test_existing_partial_output_is_not_silently_skipped(self) -> None:
        inference = _load_script("4_inference_vllm_subset_bunya")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({"dataset": ["Hover"] * 2,
                                  "claim": ["synthetic-a", "synthetic-b"],
                                  "label": [0, 1], "prompt": ["a", "b"],
                                  "evidence_order": ["multi"] * 2,
                                  "filler_length": [2048] * 2,
                                  "evidence_position": [0.0] * 2})
            prompt = root / "prompts_len2048_pos0.pqt"
            frame.to_parquet(prompt)
            output = root / "res_multi_len2048_pos0_subset_seed42.pqt"
            frame.iloc[:1].assign(pred_label=0).to_parquet(output)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                inference.run_inference_for_file(
                    None, None, None, "", prompt, root, 2, None,
                    {("hover", "synthetic-a"), ("hover", "synthetic-b")},
                    "subset_seed42", False,
                )


if __name__ == "__main__":
    unittest.main()
