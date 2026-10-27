"""Checks for side separation, missingness, denominators, and patient identity."""

import contextlib
import io
import unittest

import pandas as pd

from pathology_verification_bilateral import (
    ID_COL, INSTITUTION_COL, LNLS, NECK_DISSECTION_COL, SIDES, SUBSITE_COL,
    analyse_pathology_verification,
)


def cohort():
    values = {
        ID_COL: ["1", "1", None], INSTITUTION_COL: ["A", "B", "B"],
        SUBSITE_COL: ["C09", "C09", None],
        NECK_DISSECTION_COL: [True, False, None],
    }
    for side in SIDES:
        for lnl in LNLS:
            values[("max_llh", side, lnl)] = [False, False, False]
            values[("pathology", side, lnl)] = [None, None, None]
    values[("pathology", "ipsi", "I")] = [False, None, None]
    values[("pathology", "contra", "II")] = [True, True, None]
    values[("CT", "ipsi", "I")] = [False, None, None]
    values[("CT", "contra", "II")] = [False, True, None]
    values[("max_llh", "contra", "II")] = [False, True, None]
    # Repeated row indices must not merge patients or break boolean alignment.
    return pd.DataFrame(values, index=[7, 7, 15])


def analyse(data):
    with contextlib.redirect_stdout(io.StringIO()):
        return analyse_pathology_verification(data, output_dir=None)


class BilateralAnalysisTests(unittest.TestCase):
    def test_counts_separate_sides_and_union_patients(self):
        data = cohort()
        original = data.copy(deep=True)
        result = analyse(data)
        overall = result["main_numeric"].query("tumour_location == 'Overall'").set_index("side")
        self.assertEqual(overall.loc["ipsi", "evaluated_labels"], 15)
        self.assertEqual(overall.loc["contra", "evaluated_labels"], 14)
        self.assertEqual(overall.loc["both", "evaluated_labels"], 29)
        self.assertEqual(overall.loc["both", "missing_labels"], 1)
        self.assertEqual(overall.loc["both", "total_patients"], 3)
        self.assertEqual(overall.loc["ipsi", "patients_with_any_pathology"], 1)
        self.assertEqual(overall.loc["contra", "patients_with_any_pathology"], 2)
        self.assertEqual(overall.loc["both", "patients_with_any_pathology"], 2)
        self.assertEqual(len(result["patient_lnl"]), 30)
        self.assertIn("Unknown", result["main_numeric"]["tumour_location"].values)
        pd.testing.assert_frame_equal(data, original)

    def test_negative_pathology_counts_and_conflicts_keep_side(self):
        result = analyse(cohort())
        ipsi = result["patient_lnl"].query("side == 'ipsi' and LNL == 'I' and source_row == 0").iloc[0]
        self.assertEqual(ipsi["classification"], "Pathologically verified")
        conflicts = result["pathology_label_conflicts"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts.iloc[0]["side"], "contra")
        disagreement = result["disagreement_table"].query("tumour_location == 'Oropharynx' and side == 'contra' and LNL == 'II'").iloc[0]
        self.assertEqual(disagreement["patient_side_LNLs_with_both"], 2)
        self.assertEqual(disagreement["clinical_pathology_disagreements"], 1)
        self.assertEqual(disagreement["disagreement_percent"], 50)

    def test_nullable_labels_and_no_clinical_columns(self):
        data = cohort().drop(columns=[("CT", "ipsi", "I"), ("CT", "contra", "II")])
        data[("max_llh", "ipsi", "I")] = pd.array([pd.NA, False, True], dtype="boolean")
        data[("pathology", "ipsi", "I")] = pd.array([True, pd.NA, False], dtype="boolean")
        result = analyse(data)
        row = result["patient_lnl"].query("side == 'ipsi' and LNL == 'I' and source_row == 0").iloc[0]
        self.assertEqual(row["classification"], "Missing")
        self.assertFalse(row["training_label_disagrees_with_pathology"])
        self.assertTrue(result["disagreement_table"]["disagreement_percent"].isna().all())
        self.assertFalse(result["inferred_without_clinical_information"].empty)

    def test_all_missing_side_keeps_zero_denominator(self):
        data = cohort()
        for lnl in LNLS:
            data[("max_llh", "contra", lnl)] = None
        result = analyse(data)
        overall = result["main_summary"].query("`Tumour location` == 'Overall' and Side == 'Contralateral'").iloc[0]
        self.assertEqual(overall["Evaluated patient-side-LNL labels"], 0)
        self.assertEqual(overall["Missing labels, n"], 15)
        self.assertEqual(overall["Pathologically verified labels, n (%)"], "0 (NA)")

    def test_strings_are_normalized_without_truthiness(self):
        data = cohort()
        data[("max_llh", "ipsi", "I")] = ["False", " TRUE ", None]
        data[("pathology", "ipsi", "I")] = [0.0, 1.0, None]
        result = analyse(data)
        rows = result["patient_lnl"].query("side == 'ipsi' and LNL == 'I'")
        self.assertFalse(rows.iloc[0]["training_label"])
        self.assertTrue(rows.iloc[1]["training_label"])
        self.assertTrue(pd.isna(rows.iloc[2]["training_label"]))

    def test_missing_columns_and_invalid_values_raise(self):
        with self.assertRaisesRegex(ValueError, "Required columns"):
            analyse(cohort().drop(columns=[("max_llh", "contra", "V")]))
        data = cohort()
        data[("pathology", "contra", "V")] = ["unexpected", None, None]
        with self.assertRaisesRegex(ValueError, "Unexpected binary value"):
            analyse(data)

    def test_duplicate_patients_raise(self):
        data = cohort()
        data[INSTITUTION_COL] = ["A", "A", "B"]
        with self.assertRaisesRegex(ValueError, "duplicate institution-patient"):
            analyse(data)


if __name__ == "__main__":
    unittest.main()
