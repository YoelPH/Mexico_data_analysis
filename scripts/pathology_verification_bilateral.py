"""Audit pathology verification for LNLs I-V on both sides of a training cohort.

Notebook usage (after creating the exact training dataframe):
    from pathology_verification_bilateral import analyse_pathology_verification
    results = analyse_pathology_verification(dataset_filled)

The input is never enhanced, filtered, or modified here. As in the original
analysis, 'Pathologically verified' means a nonmissing training label with
available pathology, including negative pathology. It does not imply agreement:
conflicting labels are reported separately. 'Clinically inferred' retains the
original definition: a nonmissing training label without pathology.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display


LNLS = ["I", "II", "III", "IV", "V"]
SIDES = ["ipsi", "contra"]
SIDE_NAMES = {"ipsi": "Ipsilateral", "contra": "Contralateral", "both": "Both sides"}
CLINICAL_MODALITIES = ["CT", "FNA", "MRI", "PET", "pCT", "diagnostic_consensus"]
SUBSITE_COL = ("tumor", "core", "subsite")
ID_COL = ("patient", "core", "id")
INSTITUTION_COL = ("patient", "core", "institution")
NECK_DISSECTION_COL = ("patient", "core", "neck_dissection")
LOCATION_CODES = {
    "Oral cavity": ["C02", "C03", "C04", "C05", "C06"],
    "Oropharynx": [
        "C09", "C09.0", "C09.1", "C01", "other Oropharynx", "C05.1",
        "C10.8", "C09.9", "C09.8", "C10", "C10.2", "C10.3",
    ],
    "Hypopharynx": ["C12", "C13"],
    "Larynx": ["C32.0", "C32.1", "C32.2"],
}
LOCATION_ORDER = list(LOCATION_CODES) + ["Unknown", "Overall"]


def _boolean(series):
    """Normalize actual booleans, 0/1, and true/false strings; preserve NA."""
    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().casefold() in ("true", "false"):
            return value.strip().casefold() == "true"
        raise ValueError(f"Unexpected binary value {value!r} in column {series.name!r}")

    return series.map(convert).astype("boolean")


def inspect_training_dataframe(data):
    """Inspect the original types and both sides before any normalization."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame; received {type(data)!r}")
    if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 3:
        raise ValueError("Expected three-level MultiIndex columns: (modality, side, LNL).")
    if not data.columns.is_unique:
        raise ValueError("Duplicate column tuples make the analysis ambiguous.")
    print("Dataframe type:", type(data))
    print("Shape:", data.shape)
    print("Column index:", type(data.columns), "levels:", data.columns.nlevels)
    print("Column-level names:", data.columns.names)
    print("Row index:", type(data.index), "unique:", data.index.is_unique)
    print("All-column dtype counts:", data.dtypes.astype(str).value_counts().to_dict())
    print("Available headers:", data.columns.get_level_values(0).unique().tolist())
    print("Available second-level values:", data.columns.get_level_values(1).unique().tolist())

    columns = [
        (modality, side, lnl)
        for modality in ["max_llh", "pathology"] + CLINICAL_MODALITIES
        for side in SIDES for lnl in LNLS
    ]
    records = []
    for column in columns:
        record = dict(zip(["modality", "side", "LNL"], column))
        record["column_present"] = column in data.columns
        if record["column_present"]:
            values = data[column]
            record.update(
                dtype=str(values.dtype),
                nonmissing=int(values.notna().sum()),
                missing=int(values.isna().sum()),
                value_types=", ".join(sorted({type(v).__name__ for v in values.dropna()})),
                value_counts=repr(values.value_counts(dropna=False).to_dict()),
            )
        records.append(record)
    audit = pd.DataFrame(records)
    display(audit)

    required = [SUBSITE_COL, ID_COL, INSTITUTION_COL] + [
        (modality, side, lnl)
        for modality in ("max_llh", "pathology") for side in SIDES for lnl in LNLS
    ]
    absent = [column for column in required if column not in data.columns]
    if absent:
        raise ValueError(f"Required columns are absent (not merely missing values): {absent}")
    return audit


def _ordered(frame, columns):
    orders = {
        "tumour_location": LOCATION_ORDER, "Tumour location": LOCATION_ORDER,
        "side": SIDES + ["both"], "Side": list(SIDE_NAMES.values()), "LNL": LNLS,
    }
    frame = frame.copy()
    for column in columns:
        if column in orders:
            frame[column] = pd.Categorical(frame[column], orders[column], ordered=True)
    return frame.sort_values(columns).reset_index(drop=True)


def _label_counts(frame, groups):
    counts = frame.assign(
        evaluated=frame["training_label"].notna(),
        verified=frame["classification"].eq("Pathologically verified"),
        inferred=frame["classification"].eq("Clinically inferred"),
        missing=frame["classification"].eq("Missing"),
    ).groupby(groups, as_index=False, observed=True).agg(
        evaluated_labels=("evaluated", "sum"),
        pathologically_verified=("verified", "sum"),
        clinically_inferred=("inferred", "sum"),
        missing_labels=("missing", "sum"),
    )
    assert counts["evaluated_labels"].eq(
        counts["pathologically_verified"] + counts["clinically_inferred"]
    ).all()
    return counts


def _n_percent(n, denominator):
    if denominator == 0:
        return f"{int(n)} (NA)"
    return f"{int(n)} ({100 * n / denominator:.1f}%)"


def _format_summary(numeric, main=False):
    result = pd.DataFrame({
        "Tumour location": numeric["tumour_location"],
        "Side": numeric["side"].map(SIDE_NAMES),
    })
    if main:
        result["Total patients"] = numeric["total_patients"]
        result["Patients with any pathological LNL information, n (%)"] = [
            _n_percent(n, total) for n, total in zip(
                numeric["patients_with_any_pathology"], numeric["total_patients"]
            )
        ]
    else:
        result["LNL"] = numeric["LNL"]
    result["Evaluated patient-side-LNL labels"] = numeric["evaluated_labels"]
    for label, column in [
        ("Pathologically verified labels, n (%)", "pathologically_verified"),
        ("Clinically inferred labels, n (%)", "clinically_inferred"),
    ]:
        result[label] = [
            _n_percent(n, total) for n, total in zip(numeric[column], numeric["evaluated_labels"])
        ]
    result["Missing labels, n"] = numeric["missing_labels"]
    return _ordered(result, ["Tumour location", "Side"] + ([] if main else ["LNL"]))


def analyse_pathology_verification(dataset_filled, output_dir="bilateral_pathology_verification"):
    """Return dataframes and optionally export CSVs (output_dir=None disables export)."""
    column_audit = inspect_training_dataframe(dataset_filled)
    if dataset_filled.empty:
        raise ValueError("The training cohort is empty.")
    # Positional indexing also handles nonconsecutive or repeated source indices.
    data = dataset_filled.copy().reset_index(drop=True)
    code_to_location = {code: location for location, codes in LOCATION_CODES.items() for code in codes}
    subsite = data[SUBSITE_COL].astype("string").str.strip()
    tumour_location = subsite.map(code_to_location)
    unmapped = subsite[subsite.notna() & tumour_location.isna()].unique().tolist()
    if unmapped:
        raise ValueError(f"Unmapped ICD subsites found: {sorted(unmapped)}")
    tumour_location = tumour_location.fillna("Unknown")

    patient_id = data[ID_COL].astype("string")
    institution = data[INSTITUTION_COL].astype("string")
    duplicate_mask = patient_id.notna() & data.duplicated([INSTITUTION_COL, ID_COL], keep=False)
    print("Missing patient IDs:", int(patient_id.isna().sum()))
    print("Missing institutions:", int(institution.isna().sum()))
    print("Duplicated institution-patient ID rows:", int(duplicate_mask.sum()))
    if duplicate_mask.any():
        display(data.loc[duplicate_mask, [INSTITUTION_COL, ID_COL, SUBSITE_COL]])
        raise ValueError("Resolve duplicate institution-patient rows before counting patient-side-LNL labels.")
    # JSON encodes pairs without delimiter collisions; missing IDs use row positions.
    import json
    patient_key = pd.Series([
        json.dumps(["row", i] if pd.isna(pid) else [
            "patient", None if pd.isna(inst) else str(inst), str(pid)
        ], ensure_ascii=False)
        for i, (inst, pid) in enumerate(zip(institution, patient_id))
    ], dtype="string")

    binary = {
        (r.modality, r.side, r.LNL): _boolean(data[(r.modality, r.side, r.LNL)])
        for r in column_audit.itertuples(index=False) if r.column_present
    }
    records = []
    for side in SIDES:
        for lnl in LNLS:
            training_label = binary[("max_llh", side, lnl)]
            pathology = binary[("pathology", side, lnl)]
            clinical_columns = [(m, side, lnl) for m in CLINICAL_MODALITIES if (m, side, lnl) in binary]
            clinical_available = pd.Series(False, index=data.index)
            clinical_disagrees = pd.Series(False, index=data.index)
            for column in clinical_columns:
                diagnosis = binary[column]
                clinical_available |= diagnosis.notna()
                clinical_disagrees |= (pathology.notna() & diagnosis.notna() & pathology.ne(diagnosis)).fillna(False)
            label_conflicts = (training_label.notna() & pathology.notna() & training_label.ne(pathology)).fillna(False)
            classification = np.select(
                [training_label.isna(), pathology.notna()],
                ["Missing", "Pathologically verified"], default="Clinically inferred",
            )
            records.append(pd.DataFrame({
                "source_row": data.index, "patient_key": patient_key,
                "tumour_location": tumour_location, "subsite": data[SUBSITE_COL],
                "side": side, "LNL": lnl, "training_label": training_label,
                "pathology": pathology, "clinical_available": clinical_available,
                "clinical_disagrees_with_pathology": clinical_disagrees.astype(bool),
                "training_label_disagrees_with_pathology": label_conflicts.astype(bool),
                "classification": classification,
                "inferred_without_clinical_information": (
                    training_label.notna() & pathology.isna() & ~clinical_available
                ),
            }))
    patient_lnl = pd.concat(records, ignore_index=True)
    assert len(patient_lnl) == len(data) * len(SIDES) * len(LNLS)
    assert not patient_lnl.duplicated(["patient_key", "side", "LNL"]).any()

    # Side is retained in every clinical/pathology comparison and conflict listing.
    comparable = patient_lnl["pathology"].notna() & patient_lnl["clinical_available"]
    disagreement_table = patient_lnl.loc[comparable].groupby(
        ["tumour_location", "side", "LNL"], as_index=False, observed=True,
    ).agg(
        patient_side_LNLs_with_both=("clinical_available", "size"),
        clinical_pathology_disagreements=("clinical_disagrees_with_pathology", "sum"),
    )
    # Keep zero-denominator groups visible, with an undefined percentage.
    grid = patient_lnl[["tumour_location", "side", "LNL"]].drop_duplicates()
    disagreement_table = grid.merge(disagreement_table, how="left", validate="one_to_one")
    for column in ["patient_side_LNLs_with_both", "clinical_pathology_disagreements"]:
        disagreement_table[column] = disagreement_table[column].fillna(0).astype(int)
    disagreement_table["disagreement_percent"] = (
        100 * disagreement_table["clinical_pathology_disagreements"]
        / disagreement_table["patient_side_LNLs_with_both"].replace(0, np.nan)
    )
    disagreement_table = _ordered(disagreement_table, ["tumour_location", "side", "LNL"])
    conflict_columns = ["source_row", "patient_key", "tumour_location", "subsite", "side", "LNL", "training_label", "pathology"]
    pathology_label_conflicts = patient_lnl.loc[
        patient_lnl["training_label_disagrees_with_pathology"], conflict_columns
    ].reset_index(drop=True)

    # 'both' pools labels, but patient counts use the union of patients on either side.
    side_and_both = pd.concat([patient_lnl, patient_lnl.assign(side="both")], ignore_index=True)
    summary_data = pd.concat([
        side_and_both, side_and_both.assign(tumour_location="Overall")
    ], ignore_index=True)
    patient_pathology = summary_data.assign(has_pathology=summary_data["pathology"].notna()).groupby(
        ["tumour_location", "side", "patient_key"], as_index=False, observed=True,
    ).agg(any_pathology=("has_pathology", "any"))
    patient_counts = patient_pathology.groupby(
        ["tumour_location", "side"], as_index=False, observed=True,
    ).agg(total_patients=("patient_key", "nunique"), patients_with_any_pathology=("any_pathology", "sum"))
    main_numeric = patient_counts.merge(
        _label_counts(summary_data, ["tumour_location", "side"]),
        on=["tumour_location", "side"], validate="one_to_one",
    )
    detailed_numeric = _label_counts(patient_lnl, ["tumour_location", "side", "LNL"])
    main_summary = _format_summary(main_numeric, main=True)
    detailed_summary = _format_summary(detailed_numeric)

    classification_checks = _label_counts(patient_lnl, ["side"])
    classification_checks["expected_labels"] = len(data) * len(LNLS)
    classification_checks["counts_match"] = (
        classification_checks["evaluated_labels"] + classification_checks["missing_labels"]
    ).eq(classification_checks["expected_labels"])
    assert classification_checks["counts_match"].all()

    # Neck dissection is a patient-level field; show pathology on each side and either side.
    patient_side_pathology = patient_lnl.assign(has_pathology=patient_lnl["pathology"].notna()).groupby(
        ["patient_key", "side"], observed=True,
    )["has_pathology"].any().unstack("side").reindex(columns=SIDES)
    dissection_check = pd.DataFrame({"patient_key": patient_key})
    dissection_crosstabs = []
    if NECK_DISSECTION_COL in data.columns:
        dissection_check["neck_dissection"] = _boolean(data[NECK_DISSECTION_COL])
        for side in SIDES:
            dissection_check[f"any_pathology_{side}"] = patient_key.map(patient_side_pathology[side])
        dissection_check["any_pathology_either_side"] = dissection_check[
            [f"any_pathology_{side}" for side in SIDES]
        ].any(axis=1)
        for side, column in [("ipsi", "any_pathology_ipsi"), ("contra", "any_pathology_contra"), ("both", "any_pathology_either_side")]:
            tab = pd.crosstab(
                dissection_check["neck_dissection"].astype("string").fillna("Missing"),
                dissection_check[column], margins=True,
            ).reset_index()
            tab.insert(0, "side", side)
            dissection_crosstabs.append(tab)

    results = {
        "column_audit": column_audit,
        "patient_lnl": patient_lnl,
        "patient_pathology": patient_pathology,
        "classification_checks": classification_checks,
        "main_numeric": _ordered(main_numeric, ["tumour_location", "side"]),
        "main_summary": main_summary,
        "detailed_numeric": _ordered(detailed_numeric, ["tumour_location", "side", "LNL"]),
        "detailed_summary": detailed_summary,
        "disagreement_table": disagreement_table,
        "pathology_label_conflicts": pathology_label_conflicts,
        "inferred_without_clinical_information": patient_lnl.loc[patient_lnl["inferred_without_clinical_information"]],
        "dissection_check": dissection_check,
        "dissection_crosstabs": pd.concat(dissection_crosstabs, ignore_index=True) if dissection_crosstabs else pd.DataFrame(),
    }
    filenames = {
        "column_audit": "dataframe_column_audit.csv",
        "classification_checks": "classification_checks_by_side.csv",
        "main_summary": "pathology_verification_summary.csv",
        "detailed_summary": "pathology_verification_by_location_side_and_lnl.csv",
        "disagreement_table": "clinical_pathology_disagreements.csv",
        "pathology_label_conflicts": "max_llh_pathology_conflicts.csv",
        "inferred_without_clinical_information": "inferred_without_clinical_information.csv",
        "dissection_check": "neck_dissection_by_side.csv",
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, filename in filenames.items():
            results[name].to_csv(output / filename, index=False)
        print("CSV output directory:", output.resolve())
    for title, name in [
        ("CLASSIFICATION CHECKS BY SIDE", "classification_checks"),
        ("MAIN SUMMARY", "main_summary"),
        ("DETAILED LOCATION, SIDE AND LNL SUMMARY", "detailed_summary"),
        ("CLINICAL-PATHOLOGICAL DISAGREEMENTS", "disagreement_table"),
        ("MAX_LLH-PATHOLOGY CONFLICTS", "pathology_label_conflicts"),
        ("NECK-DISSECTION CHECK", "dissection_crosstabs"),
    ]:
        print("\n" + title)
        display(results[name])
    print("Inferred labels without listed clinical information:", len(results["inferred_without_clinical_information"]))
    print("'Pathologically verified' records pathology availability; see the separate conflict table for agreement.")
    return results
