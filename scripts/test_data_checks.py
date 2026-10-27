"""Behavioral checks for the shared missingness and comparison rules."""

import unittest

import pandas as pd

from scripts import data_checks as checks


def cohort(*overrides, complete_ct=False):
    records = []
    for i, changes in enumerate(overrides):
        record = {
            ("patient", "core", "id"): f"P{i + 1}",
            ("patient", "core", "institution"): "Synthetic centre",
            ("patient", "core", "neck_dissection"): False,
            ("patient", "core", "n_stage"): "2a",
        }
        if complete_ct:
            record.update({("CT", side, level): False for side, level in checks.PAIR_COLUMNS})
        record.update(changes)
        records.append(record)
    result = pd.DataFrame(records)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return checks.normalize_source(result)


class DataChecksTest(unittest.TestCase):
    def test_pathology_includes_negative_sublevel_and_unknown_dissection(self):
        raw = cohort(
            {("pathology", "ipsi", "IIa"): True},
            {("pathology", "contra", "III"): False,
             ("patient", "core", "neck_dissection"): None},
            {("pathology", "ipsi", "II"): True,
             ("patient", "core", "neck_dissection"): True},
            {("pathology", "ipsi", "II"): "unknown"},
        )
        result = checks.pathology_without_dissection(raw)
        self.assertEqual(result["patient_id"].tolist(), ["P1", "P2"])
        self.assertIn("IIa=positive", result.iloc[0]["pathology_observations"])
        self.assertIn("III=negative", result.iloc[1]["pathology_observations"])

    def test_missing_and_absent_modalities_have_full_denominators(self):
        raw = cohort({("CT", "ipsi", "I"): False}, {})
        overview = checks.missing_overview(raw).set_index("modality")
        self.assertEqual(overview.loc["CT", "observed_pairs"], 1)
        self.assertEqual(overview.loc["CT", "missing_pairs"], 23)
        self.assertEqual(overview.loc["pCT", "columns_present"], 0)
        self.assertEqual(overview.loc["pCT", "missing_pairs"], 24)

    def test_hierarchy_requires_sufficient_negative_information(self):
        raw = cohort(
            {("CT", "ipsi", "II"): False, ("CT", "ipsi", "IIa"): True},
            {("CT", "ipsi", "II"): True, ("CT", "ipsi", "IIa"): False},
            {("CT", "ipsi", "II"): True, ("CT", "ipsi", "IIa"): False,
             ("CT", "ipsi", "IIb"): False},
        )
        result = checks.hierarchy_conflicts(raw)
        self.assertEqual(result["patient_id"].tolist(), ["P1", "P3"])
        empty = checks.hierarchy_conflicts(cohort({}))
        self.assertTrue(empty.empty)
        self.assertIn("row", empty.columns)

    def test_reconstruction_excludes_recorded_consensus_and_preserves_unknown(self):
        raw = cohort({("diagnostic_consensus", "ipsi", "III"): True})
        raw.index = pd.Index([42])
        results, _ = checks.reconstruct(raw)
        for table in results.values():
            self.assertEqual(table.index.tolist(), [42])
            self.assertTrue(table.isna().all().all())

    def test_pathology_changes_estimate_without_modifying_source(self):
        raw = cohort({("CT", "ipsi", "III"): True,
                      ("pathology", "ipsi", "III"): False,
                      ("diagnostic_consensus", "ipsi", "III"): False}, complete_ct=True)
        raw.index = pd.Index([37])
        before = raw.copy(deep=True)
        results, _ = checks.reconstruct(raw)
        self.assertTrue(results["max_llh_without_pathology"].loc[37, ("ipsi", "III")])
        self.assertFalse(results["max_llh_with_pathology"].loc[37, ("ipsi", "III")])
        pd.testing.assert_frame_equal(raw, before)
        raw[("diagnostic_consensus", "ipsi", "III")] = True
        changed_consensus, _ = checks.reconstruct(raw)
        for name in results:
            pd.testing.assert_frame_equal(results[name], changed_consensus[name])

    def test_fna_is_retained_without_pathology(self):
        raw = cohort({("FNA", "contra", "III"): True})
        results, _ = checks.reconstruct(raw)
        self.assertTrue(results["max_llh_without_pathology"].loc[0, ("contra", "III")])
        self.assertTrue(pd.isna(results["max_llh_without_pathology"].loc[0, ("ipsi", "III")]))

    def test_only_known_pairs_count_and_disagreements_are_not_duplicated(self):
        raw = cohort({("CT", "ipsi", "I"): True, ("pathology", "ipsi", "I"): False},
                     {("CT", "ipsi", "I"): False})
        sources = {name: checks.observations(raw, name) for name in ["CT", "pathology", "MRI"]}
        summary = checks.comparison_summary(sources, [("CT", "pathology"), ("CT", "MRI")])
        self.assertEqual(summary.iloc[0]["compared_pairs"], 1)
        self.assertEqual(summary.iloc[0]["left_positive_right_negative"], 1)
        self.assertEqual(summary.iloc[0]["patients_affected"], 1)
        self.assertEqual(summary.iloc[1]["compared_pairs"], 0)
        self.assertTrue(pd.isna(summary.iloc[1]["agreement_pct"]))
        details = checks.disagreement_details(raw, sources, [("CT", "pathology"), ("pathology", "CT")])
        self.assertEqual(len(details), 1)

    def test_matched_pair_mask_and_index_validation(self):
        raw = cohort({("CT", "ipsi", "I"): True, ("MRI", "ipsi", "I"): False},
                     {("CT", "ipsi", "I"): False, ("MRI", "ipsi", "I"): False})
        sources = {name: checks.observations(raw, name) for name in ["CT", "MRI"]}
        mask = pd.DataFrame(False, index=raw.index, columns=checks.PAIR_COLUMNS)
        mask.loc[0, ("ipsi", "I")] = True
        summary = checks.comparison_summary(sources, [("CT", "MRI")], mask=mask)
        self.assertEqual(summary.iloc[0]["compared_pairs"], 1)
        self.assertEqual(summary.iloc[0]["disagreements"], 1)
        sources["MRI"].index = [10, 11]
        with self.assertRaises(ValueError):
            checks.comparison_summary(sources, [("CT", "MRI")])

    def test_staging_distinguishes_complete_partial_and_absent_observations(self):
        negative = {("CT", side, level): False for side, level in checks.PAIR_COLUMNS}
        raw = cohort(
            {**negative, ("tumor", "core", "t_stage_prefix"): "c"},
            {("CT", "ipsi", "I"): False},
            {},
            {("patient", "core", "n_stage"): "N0", ("CT", "contra", "III"): True},
        )
        result = checks.n_stage_comparison(raw, {"CT": checks.observations(raw, "CT")})
        self.assertEqual(result.iloc[0]["stage_basis"], "unconfirmed")
        self.assertEqual(result.iloc[0]["assessment"], "Review: N+ with all checked levels negative")
        self.assertIn("incomplete coverage", result.iloc[1]["assessment"])
        self.assertEqual(result.iloc[2]["assessment"], "No involvement observations")
        self.assertEqual(result.iloc[3]["assessment"], "Review: N0 with positive involvement")
        self.assertEqual(result.iloc[3]["contra_positive_levels"], "III")

    def test_n_stage_prefix_has_its_own_basis(self):
        raw = cohort({("patient", "core", "n_stage"): "pN1"},
                     {("patient", "core", "n_stage"): "pN1",
                      ("patient", "core", "n_stage_prefix"): "c"})
        result = checks.n_stage_comparison(raw, {"CT": checks.observations(raw, "CT")})
        self.assertEqual(result.iloc[0]["stage_basis"], "pathological")
        self.assertEqual(result.iloc[1]["stage_basis"], "conflicting/invalid N prefix")


if __name__ == "__main__":
    unittest.main()
