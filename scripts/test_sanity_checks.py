"""Behavioral checks for missingness, staging context, counts, and patient identity."""

import unittest
from pathlib import Path

import pandas as pd

from sanity_checks import MAIN_LEVELS, INCOMPLETE, REVIEW, run_sanity_checks


def cohort(*overrides):
    records = []
    for i, changes in enumerate(overrides):
        record = {
            ("patient", "core", "id"): f"P{i + 1}",
            ("patient", "core", "institution"): "Test centre",
            ("patient", "core", "n_stage"): "N0",
            ("tumor", "core", "subsite"): "C09",
        }
        for side in ("ipsi", "contra"):
            for level in MAIN_LEVELS:
                record[("diagnostic_consensus", side, level)] = False
        record.update(changes)
        records.append(record)
    df = pd.DataFrame(records)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def codes(df, **kwargs):
    return set(run_sanity_checks(df, **kwargs)["check"])


class SanityChecksTest(unittest.TestCase):
    def test_consistent_n0_is_clean_and_input_is_unchanged(self):
        df = cohort({})
        before = df.copy(deep=True)
        for mode in ("full", "reduced"):
            with self.subTest(mode=mode):
                self.assertTrue(run_sanity_checks(df, mode=mode).empty)
                pd.testing.assert_frame_equal(df, before)

    def test_positive_stage_strings_are_not_lost(self):
        for stage in (1, 2.0, "2a", "2b", "2c", "N3b", "cN2", "pN2"):
            with self.subTest(stage=stage):
                found = codes(cohort({("patient", "core", "n_stage"): stage}))
                expected = ("n_positive_without_localization" if stage == "pN2"
                            else "n_positive_all_regions_negative")
                self.assertIn(expected, found)
                self.assertNotIn("invalid_n_stage", found)

    def test_missing_is_not_negative(self):
        df = cohort({("patient", "core", "n_stage"): "2b",
                     ("diagnostic_consensus", "contra", "VII"): pd.NA})
        found = codes(df)
        self.assertIn("n_positive_without_localization", found)
        self.assertNotIn("n_positive_all_regions_negative", found)

    def test_unknown_boolean_strings_are_missing(self):
        df = cohort({("patient", "core", "n_stage"): "N1",
                     ("diagnostic_consensus", "contra", "VII"): "unknown"})
        self.assertIn("n_positive_without_localization", codes(df))

    def test_positive_sublevel_or_vii_prevents_no_positive_flag(self):
        for level in ("IIa", "VII", "VI"):
            with self.subTest(level=level):
                df = cohort({("patient", "core", "n_stage"): "2b",
                             ("CT", "contra", level): True})
                self.assertFalse(any(c.startswith("n_positive_") for c in codes(df)))

    def test_n0_with_positive_is_flagged(self):
        df = cohort({("CT", "ipsi", "IIa"): "true"})
        issue = run_sanity_checks(df).query("check == 'n0_with_positive_involvement'").iloc[0]
        self.assertEqual(issue["category"], REVIEW)
        self.assertIn("IIa", issue["details"])

    def test_clinical_stage_does_not_use_pathology_or_separate_imaging(self):
        df = cohort({("patient", "core", "n_stage"): "cN0",
                     ("pathology", "ipsi", "II"): True,
                     ("CT", "ipsi", "II"): True})
        self.assertNotIn("n0_with_positive_involvement", codes(df))

    def test_pathological_stage_does_not_use_clinical_positives(self):
        df = cohort({("patient", "core", "n_stage"): "pN0",
                     ("diagnostic_consensus", "ipsi", "II"): True})
        self.assertNotIn("n0_with_positive_involvement", codes(df))

    def test_known_positive_counts_support_pathological_stage(self):
        df = cohort({("patient", "core", "n_stage"): "pN1",
                     ("positive_dissected", "ipsi", "II"): 2,
                     ("total_dissected", "ipsi", "II"): 4})
        self.assertFalse(any(c.startswith("n_positive_") for c in codes(df)))

    def test_sublevel_negatives_need_both_children_in_same_modality(self):
        df = cohort({("patient", "core", "n_stage"): "N1",
                     ("diagnostic_consensus", "ipsi", "II"): None,
                     ("CT", "ipsi", "IIa"): False,
                     ("MRI", "ipsi", "IIb"): False})
        self.assertIn("n_positive_without_localization", codes(df))
        df[("CT", "ipsi", "IIb")] = False
        self.assertIn("n_positive_all_regions_negative", codes(df))

    def test_no_data_and_missing_side_are_not_flagged(self):
        df = cohort({})
        cols = [c for c in df if c[0] == "diagnostic_consensus"]
        df[cols] = None
        self.assertTrue(run_sanity_checks(df).empty)
        self.assertEqual(codes(df, mode="full"), {"no_involvement_data"})
        df[("CT", "ipsi", "II")] = False
        self.assertTrue(run_sanity_checks(df).empty)
        self.assertEqual(codes(df, mode="full"), {"side_without_involvement_data"})

    def test_missing_or_unknown_n_stage_is_not_flagged_or_treated_as_n0(self):
        for stage in (None, pd.NA, "", "unknown", "NX", "cNX", -1):
            with self.subTest(stage=stage):
                df = cohort({("patient", "core", "n_stage"): stage,
                             ("CT", "ipsi", "II"): True})
                self.assertTrue(run_sanity_checks(df).empty)
                self.assertEqual(codes(df, mode="full"), {"n_stage_unavailable"})
        df = df.drop(columns=[("patient", "core", "n_stage")])
        self.assertTrue(run_sanity_checks(df).empty)
        self.assertEqual(codes(df, mode="full"), {"n_stage_unavailable"})

    def test_node_count_and_pathology_checks(self):
        df = cohort({("total_dissected", "ipsi", "II"): 1,
                     ("positive_dissected", "ipsi", "II"): 2,
                     ("pathology", "ipsi", "II"): False})
        found = codes(df)
        self.assertNotIn("positive_exceeds_total", found)
        self.assertIn("positive_count_negative_pathology", found)
        self.assertEqual(codes(df, mode="full"), found | {"positive_exceeds_total"})

    def test_counts_do_not_compare_different_levels_or_sum_parent_and_children(self):
        df = cohort({("patient", "core", "n_stage"): 1,
                     ("total_dissected", "ipsi", "I"): 3,
                     ("positive_dissected", "ipsi", "I"): 2,
                     ("total_dissected", "ipsi", "Ia"): 2,
                     ("positive_dissected", "ipsi", "Ia"): 2,
                     ("total_dissected", "contra", "I"): 0})
        self.assertNotIn("positive_exceeds_total", codes(df, mode="full"))

    def test_pathology_positive_with_sampled_zero_positives(self):
        df = cohort({("total_dissected", "ipsi", "II"): 5,
                     ("positive_dissected", "ipsi", "II"): 0,
                     ("pathology", "ipsi", "II"): True})
        for mode in ("full", "reduced"):
            self.assertIn("zero_positive_count_positive_pathology", codes(df, mode=mode))

    def test_optional_node_counts_are_not_required(self):
        cases = [
            {("positive_dissected", "ipsi", "II"): 2},
            {("pathology", "ipsi", "II"): True},
            {("pathology", "ipsi", "II"): True,
             ("total_dissected", "ipsi", "II"): 5,
             ("positive_dissected", "ipsi", "II"): None},
            {("pathology", "ipsi", "II"): True,
             ("total_dissected", "ipsi", "II"): 0,
             ("positive_dissected", "ipsi", "II"): 0},
            {("pathology", "ipsi", "II"): True,
             ("total_dissected", "ipsi", "II"): None,
             ("positive_dissected", "ipsi", "II"): 0},
            {("pathology", "ipsi", "II"): True,
             ("total_dissected", "ipsi", "II"): "unknown",
             ("positive_dissected", "ipsi", "II"): "unknown"},
            {("pathology", "ipsi", "II"): False},
            {("pathology", "ipsi", "II"): False,
             ("total_dissected", "ipsi", "II"): 0},
        ]
        for observations in cases:
            with self.subTest(observations=observations):
                df = cohort({("patient", "core", "n_stage"): "cN0",
                             **observations})
                self.assertTrue(run_sanity_checks(df).empty)
                self.assertNotIn("zero_positive_count_positive_pathology", codes(df, mode="full"))

    def test_invalid_counts_do_not_become_zeros(self):
        df = cohort({("total_dissected", "ipsi", "II"): 3,
                     ("positive_dissected", "ipsi", "II"): -1,
                     ("pathology", "ipsi", "II"): True})
        found = codes(df)
        self.assertIn("invalid_node_count", found)
        self.assertNotIn("zero_positive_count_positive_pathology", found)

    def test_dissection_checks(self):
        df = cohort({("patient", "core", "neck_dissection"): False,
                     ("total_dissected", "ipsi", "II"): 4})
        self.assertIn("dissection_flag_vs_counts", codes(df))
        df = cohort({("patient", "core", "neck_dissection"): True})
        self.assertIn("dissection_without_results", codes(df))

    def test_duplicate_ids_are_scoped_to_centre(self):
        same_id = {("patient", "core", "id"): "001"}
        df = cohort(same_id, {**same_id, ("patient", "core", "institution"): "Other centre"})
        self.assertNotIn("duplicate_patient", codes(df))
        df = cohort(same_id, same_id)
        self.assertEqual((run_sanity_checks(df)["check"] == "duplicate_patient").sum(), 2)

    def test_missing_ids_do_not_get_grouped_as_duplicates(self):
        df = cohort({("patient", "core", "id"): None}, {("patient", "core", "id"): None})
        found = codes(df)
        self.assertIn("missing_identity", found)
        self.assertNotIn("duplicate_patient", found)

    def test_invalid_values_are_flagged_but_dates_and_chronology_are_ignored(self):
        df = cohort({("CT", "ipsi", "II"): "maybe",
                     ("patient", "core", "n_stage"): "N7",
                     ("tumor", "core", "volume"): -2,
                     ("CT", "core", "date"): "not a valid date",
                     ("patient", "core", "diagnose_date"): "2025-10-12",
                     ("outcome", "core", "date_recurrence"): "2024-10-12",
                     ("outcome", "core", "date_last_follow_up"): "2023-10-12"})
        found = codes(df)
        self.assertEqual(found, {"invalid_boolean", "invalid_n_stage", "invalid_numeric"})
        self.assertEqual(codes(df, mode="full"), found | {
            "invalid_date", "recurrence_before_diagnosis", "follow_up_before_diagnosis",
            "recurrence_after_follow_up",
        })

    def test_empty_input_preserves_output_columns(self):
        result = run_sanity_checks(cohort({}).iloc[:0])
        self.assertTrue(result.empty)
        self.assertIn("details", result.columns)

    def test_explicit_basis_and_conflicting_prefixes(self):
        df = cohort({("pathology", "ipsi", "II"): True})
        self.assertNotIn("n0_with_positive_involvement", codes(df, stage_basis="clinical"))
        self.assertIn("n0_with_positive_involvement", codes(df, stage_basis="pathological"))
        df[("patient", "core", "n_stage")] = "cN0"
        df[("patient", "core", "n_stage_prefix")] = "p"
        self.assertIn("n_stage_prefix_conflict", codes(df))



class CheckModesTest(unittest.TestCase):
    def test_full_restores_all_omitted_checks_and_reduced_is_the_default(self):
        no_involvement = {(mod, side, level): None for mod in ("diagnostic_consensus",)
                          for side in ("ipsi", "contra") for level in MAIN_LEVELS}
        missing_side = {("diagnostic_consensus", "contra", level): None for level in MAIN_LEVELS}
        df = cohort(
            {**no_involvement, ("patient", "core", "n_stage"): None},
            missing_side,
            {("patient", "core", "n_stage"): "cN0",
             ("total_dissected", "ipsi", "II"): 1,
             ("positive_dissected", "ipsi", "II"): 2,
             ("pathology", "ipsi", "II"): False,
             ("total_dissected", "contra", "III"): 0,
             ("pathology", "contra", "III"): False,
             ("positive_dissected", "ipsi", "IV"): 1,
             ("CT", "core", "date"): "not a valid date",
             ("patient", "core", "diagnose_date"): "2025-10-12",
             ("outcome", "core", "date_recurrence"): "2024-10-12",
             ("outcome", "core", "date_last_follow_up"): "2023-10-12"},
        )
        before = df.copy(deep=True)
        full = run_sanity_checks(df, mode="full")
        reduced = run_sanity_checks(df, mode="reduced")
        full_only = {
            "invalid_date", "recurrence_before_diagnosis", "follow_up_before_diagnosis",
            "recurrence_after_follow_up", "n_stage_unavailable", "no_involvement_data",
            "side_without_involvement_data", "positive_exceeds_total",
            "positive_count_without_total", "negative_pathology_without_sample",
        }
        self.assertEqual(set(full["check"]) - set(reduced["check"]), full_only)
        self.assertEqual(reduced["check"].tolist(), ["positive_count_negative_pathology"])
        expected_reduced = full.loc[~full["check"].isin(full_only)].reset_index(drop=True)
        pd.testing.assert_frame_equal(reduced, expected_reduced)
        pd.testing.assert_frame_equal(run_sanity_checks(df), reduced)
        pd.testing.assert_frame_equal(df, before)
        self.assertEqual(full.attrs["mode"], "full")
        self.assertEqual(reduced.attrs["mode"], "reduced")

    def test_invalid_mode_is_rejected(self):
        for mode in (None, "", "FULL", "typo"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "mode"):
                run_sanity_checks(cohort({}), mode=mode)

    def test_full_mode_groups_unsampled_negative_levels_per_patient(self):
        df = cohort({("total_dissected", "ipsi", "I"): 0,
                     ("total_dissected", "contra", "II"): 0,
                     ("pathology", "ipsi", "I"): False,
                     ("pathology", "contra", "II"): False})
        issues = run_sanity_checks(df, mode="full")
        self.assertEqual(issues["check"].tolist(), ["negative_pathology_without_sample"])
        self.assertEqual(issues.iloc[0]["category"], REVIEW)
        self.assertIn("ipsilateral: I", issues.iloc[0]["details"])
        self.assertIn("contralateral: II", issues.iloc[0]["details"])
        self.assertTrue(run_sanity_checks(df, mode="reduced").empty)

    def test_full_mode_iso_and_excel_dates_are_unambiguous(self):
        df = cohort({("patient", "core", "diagnose_date"): "2025-01-12",
                     ("outcome", "core", "date_last_follow_up"): "2025-02-01",
                     ("CT", "core", "date"): 45000})
        for dayfirst in (True, False):
            with self.subTest(dayfirst=dayfirst):
                self.assertTrue(run_sanity_checks(df, mode="full", dayfirst=dayfirst).empty)

    def test_full_mode_dayfirst_can_be_selected(self):
        df = cohort({("patient", "core", "diagnose_date"): "01/02/2025",
                     ("outcome", "core", "date_last_follow_up"): "02/01/2025"})
        self.assertEqual(codes(df, mode="full"), {"follow_up_before_diagnosis"})
        self.assertTrue(run_sanity_checks(df, mode="full", dayfirst=False).empty)

    def test_full_mode_date_placeholders_are_unavailable_not_invalid(self):
        for value in ("not done", "Not done", "NO", False, "no les damos seguimiento", "not applicable"):
            with self.subTest(value=value):
                df = cohort({("CT", "core", "date"): value})
                self.assertTrue(run_sanity_checks(df, mode="full").empty)


class UpdatedCohortSmokeTest(unittest.TestCase):
    def test_current_five_updated_workbooks(self):
        folder = Path(__file__).resolve().parents[1] / "updated_datasets"
        paths = [folder / f"{name}_update.xlsx" for name in ("ABC", "INC", "MID", "UEO", "XXI")]
        if not all(path.exists() for path in paths):
            self.skipTest("The five updated workbooks are not available.")
        raw = pd.concat([pd.read_excel(path, header=[0, 1, 2]) for path in paths], ignore_index=True)
        before = raw.copy(deep=True)
        for mode in ("full", "reduced"):
            with self.subTest(mode=mode):
                result = run_sanity_checks(raw, mode=mode)
                pd.testing.assert_frame_equal(raw, before)
                self.assertTrue(set(result["row"]).issubset(set(raw.index)))
                self.assertEqual(result.attrs["missing_columns"], [])
                self.assertEqual(result.attrs["mode"], mode)
                print(f"\nCurrent cohort ({mode}): {len(raw)} rows; {len(result)} review observations.")
                print(result.groupby(["check", "category"]).size().to_string())


if __name__ == "__main__":
    unittest.main()
