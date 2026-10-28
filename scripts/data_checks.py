"""Shared notebook analysis. Start with analyse_dataset(), then follow its helpers."""

import numpy as np
import pandas as pd
import lydata  # Registers the pandas accessor.
from lydata.utils import get_default_modalities

from .sanity_checks import MAIN_LEVELS, SIDES, SUBDIVISIONS, _boolean, _stage, run_sanity_checks


ALL_LEVELS = ("I", "Ia", "Ib", "II", "IIa", "IIb", "III", "IV", "V", "VII")
MODALITIES = ("diagnostic_consensus", "pathology", "CT", "MRI", "PET", "FNA", "pCT")
WITHOUT_PATHOLOGY = ("CT", "MRI", "PET", "pCT", "FNA")
PAIR_COLUMNS = pd.MultiIndex.from_product([SIDES, MAIN_LEVELS], names=["side", "level"])


def analyse_dataset(data):
    """Run the four notebook analyses and return their tables. Input stays unchanged."""
    raw = normalize_source(data)
    hierarchy = hierarchy_conflicts(raw)
    sanity = run_sanity_checks(raw, mode="reduced")
    sources, consensus = _consensus_tables(raw)

    return {
        "raw": raw,
        "pathology_without_neck_dissection": pathology_without_dissection(raw),
        "missing_observations": missing_overview(raw),
        "hierarchy_issues": hierarchy,
        "sanity_issues": sanity,
        "consistency_summary": pd.DataFrame({
            "check": ["Main level vs sublevels", "Reduced sanity checks"],
            "observations": [len(hierarchy), len(sanity)],
            "patients_affected": [hierarchy["row"].nunique(), sanity["row"].nunique()],
        }),
        **consensus,
        **_stage_tables(raw, sources),
    }


def normalize_source(data):
    """Normalize workbook placeholders and booleans, retaining an independent copy."""
    if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 3:
        raise ValueError("Expected three-level workbook headers.")
    if not data.columns.is_unique or not data.index.is_unique:
        raise ValueError("Resolve duplicate columns or row indices first.")
    if data.empty:
        raise ValueError("The workbook contains no patient rows.")

    def convert(value):
        if isinstance(value, str):
            text = value.strip().casefold()
            if text in {"", "unknown", "not recorded", "nan", "none", "<na>", "na", "n/a"}:
                return np.nan
            if text in {"true", "false"}:
                return text == "true"
        return value

    return data.map(convert)


def _valid_observation(value):
    """Invalid source values remain in raw data and are flagged by sanity checks."""
    try:
        return _boolean(value)
    except (ValueError, TypeError):
        return None


def observations(raw, modality, levels=MAIN_LEVELS):
    """A fixed grid of nullable booleans; absent columns remain unknown."""
    columns = pd.MultiIndex.from_product([[modality], SIDES, levels])
    result = raw.reindex(columns=columns).copy()
    result.columns = columns.droplevel(0).set_names(["side", "level"])
    return result.map(_valid_observation).astype("boolean")


def patient_context(raw):
    result = pd.DataFrame(index=raw.index)
    result["patient_id"] = raw.get(("patient", "core", "id"), pd.NA)
    for name in ("neck_dissection", "n_stage", "n_stage_prefix", "comments"):
        result[name] = raw.get(("patient", "core", name), pd.NA)
    return result


def pathology_without_dissection(raw):
    """Include known positive OR negative pathology, including sublevel-only records."""
    pathology = observations(raw, "pathology", ALL_LEVELS)
    context = patient_context(raw)
    neck = context["neck_dissection"].map(_valid_observation).astype("boolean")
    mask = pathology.notna().any(axis=1) & ~neck.fillna(False)
    result = context.loc[mask, ["patient_id", "neck_dissection", "comments"]].copy()
    result.insert(2, "pathology_observations", [
        "; ".join(f"{side} {level}={'positive' if value else 'negative'}"
                  for (side, level), value in row.items() if pd.notna(value))
        for _, row in pathology.loc[mask].iterrows()
    ])
    return result.reset_index(drop=True)


def missing_overview(raw):
    records = []
    for modality in MODALITIES:
        values = observations(raw, modality)
        observed = int(values.notna().to_numpy().sum())
        records.append({
            "modality": modality,
            "columns_present": sum((modality, side, level) in raw.columns
                                   for side, level in PAIR_COLUMNS),
            "patients_with_data": int(values.notna().any(axis=1).sum()),
            "observed_pairs": observed,
            "missing_pairs": values.size - observed,
            "coverage_pct": 100 * observed / values.size if values.size else np.nan,
        })
    return pd.DataFrame(records)


def hierarchy_conflicts(raw, *, modalities=MODALITIES, subdivisions=SUBDIVISIONS):
    """Check known parent/child contradictions for the requested sources and levels."""
    levels = tuple(dict.fromkeys(
        level for parent, children in subdivisions.items() for level in (parent, *children)
    ))
    records = []
    for modality in modalities:
        values = observations(raw, modality, levels)
        for index, row in values.iterrows():
            for side in SIDES:
                for parent, children in subdivisions.items():
                    p = row[(side, parent)]
                    child_values = [row[(side, child)] for child in children]
                    any_positive = any(pd.notna(v) and bool(v) for v in child_values)
                    all_negative = all(pd.notna(v) and not bool(v) for v in child_values)
                    expected = True if any_positive else False if all_negative else None
                    if pd.notna(p) and expected is not None and p != expected:
                        records.append({
                            "row": index,
                            "patient_id": raw.loc[index].get(("patient", "core", "id"), pd.NA),
                            "modality": modality, "side": side, "level": parent,
                            "parent_value": bool(p),
                            "sublevels": ", ".join(f"{c}={v}" for c, v in zip(children, child_values)),
                        })
    return pd.DataFrame(records, columns=[
        "row", "patient_id", "modality", "side", "level", "parent_value", "sublevels",
    ])


def reconstruct(raw):
    """Two max_llh estimates with identical settings except inclusion of pathology.

    Recorded diagnostic_consensus is excluded from both estimates. FNA is retained.
    combine() may infer I/II parent or child labels; source observations are untouched.
    """
    defaults = get_default_modalities()
    available = {column[0] for column in raw.columns if column[1] in SIDES}
    results = {}
    hierarchy = {}
    all_columns = pd.MultiIndex.from_product([SIDES, ALL_LEVELS], names=["side", "level"])
    for name, names in [
        ("max_llh_without_pathology", WITHOUT_PATHOLOGY),
        ("max_llh_with_pathology", (*WITHOUT_PATHOLOGY, "pathology")),
    ]:
        configs = {m: defaults[m] for m in names if m in available}
        if configs:
            # Exclude invalid binary entries from inference; sanity checks report
            # their original values. Dates and metadata are not diagnostic inputs.
            columns = [c for c in raw.columns if c[0] in configs and c[1] in SIDES]
            inputs = raw[columns].map(_valid_observation)
            combined = inputs.ly.combine(
                modalities=configs, method="max_llh",
                subdivisions={"I": ["a", "b"], "II": ["a", "b"]},
            )
            # lydata returns a positional RangeIndex even if source indices differ.
            combined.index = raw.index
            results[name] = combined.reindex(columns=PAIR_COLUMNS).astype("boolean")
            before = inputs.ly.combine(modalities=configs, method="max_llh", subdivisions={})
            before.index = raw.index
            hierarchy[name] = {
                "before": before.reindex(columns=all_columns).astype("boolean"),
                "after": combined.reindex(columns=all_columns).astype("boolean"),
            }
        else:
            results[name] = pd.DataFrame(pd.NA, index=raw.index, columns=PAIR_COLUMNS, dtype="boolean")
            empty = pd.DataFrame(pd.NA, index=raw.index, columns=all_columns, dtype="boolean")
            hierarchy[name] = {"before": empty.copy(), "after": empty.copy()}
    settings = pd.DataFrame([
        {"modality": name, "available": name in available,
         "sensitivity": defaults[name].sens, "specificity": defaults[name].spec}
        for name in (*WITHOUT_PATHOLOGY, "pathology")
    ])
    return results, settings, hierarchy


def _symbol(value):
    return "—" if pd.isna(value) else "+" if bool(value) else "−"


def _raw_modality_evidence(row, side, level, subdivisions=SUBDIVISIONS):
    """Include known parent and sublevel observations, even without a parent label."""
    evidence = []
    for modality in WITHOUT_PATHOLOGY:
        values = []
        for source_level in (level, *subdivisions.get(level, ())):
            value = _valid_observation(row.get((modality, side, source_level)))
            if value is not None:
                values.append(f"{source_level} {_symbol(value)}")
        if values:
            evidence.append(f"{modality} {', '.join(values)}")
    return "; ".join(evidence) or "—"


def _hierarchy_change_text(index, side, parent, hierarchy):
    if parent not in SUBDIVISIONS:
        return "—"
    changes = []
    for name, stages in hierarchy.items():
        before = stages["before"].loc[index, (side, parent)]
        after = stages["after"].loc[index, (side, parent)]
        if _symbol(before) == _symbol(after):
            continue
        children = ", ".join(
            f"{child} {_symbol(stages['after'].loc[index, (side, child)])}"
            for child in SUBDIVISIONS[parent]
        )
        label = "Without pathology" if name == "max_llh_without_pathology" else "With pathology"
        changes.append(f"{label}: {parent} {_symbol(before)} → {_symbol(after)} ({children})")
    return "; ".join(changes) or "—"


def _paired(left, right, mask=None):
    if not left.index.equals(right.index) or not left.columns.equals(right.columns):
        raise ValueError("Comparison grids must have identical patients and columns.")
    known = left.notna() & right.notna()
    if mask is not None:
        if not mask.index.equals(left.index) or not mask.columns.equals(left.columns):
            raise ValueError("Comparison mask must match the observation grid.")
        known &= mask.fillna(False)
    different = known & left.ne(right).fillna(False)
    return known, different


def comparison_summary(sources, comparisons, mask=None):
    records = []
    for left_name, right_name in comparisons:
        left, right = sources[left_name], sources[right_name]
        known, different = _paired(left, right, mask)
        n = int(known.to_numpy().sum())
        disagreements = int(different.to_numpy().sum())
        records.append({
            "comparison": f"{left_name} vs {right_name}", "compared_pairs": n,
            "disagreements": disagreements,
            "patients_affected": int(different.any(axis=1).sum()),
            "left_positive_right_negative": int((different & left.eq(True).fillna(False)).to_numpy().sum()),
            "left_negative_right_positive": int((different & right.eq(True).fillna(False)).to_numpy().sum()),
            "agreement_pct": 100 * (n - disagreements) / n if n else np.nan,
        })
    return pd.DataFrame(records)


def disagreement_details(raw, sources, comparisons, *, hierarchy=None,
                         subdivisions=SUBDIVISIONS):
    """One evidence row per disputed patient/side/level, with no duplicate pairs."""
    reasons = {}
    for left_name, right_name in comparisons:
        _, different = _paired(sources[left_name], sources[right_name])
        for position, row in enumerate(different.to_numpy(dtype=bool)):
            for column_position in np.flatnonzero(row):
                reasons.setdefault((position, int(column_position)), []).append(f"{left_name} vs {right_name}")
    context = patient_context(raw)
    pair_columns = next(iter(sources.values())).columns if sources else ()
    records = []
    for (position, column_position), pairs in sorted(reasons.items()):
        side, level = pair_columns[column_position]
        records.append({
            "row": raw.index[position],
            "patient_id": context.iloc[position]["patient_id"], "side": side, "level": level,
            **{name: values.iloc[position, column_position] for name, values in sources.items()},
            "raw_modalities": _raw_modality_evidence(raw.iloc[position], side, level, subdivisions),
            "hierarchy_changes": _hierarchy_change_text(raw.index[position], side, level, hierarchy or {}),
            "disagreements": "; ".join(pairs), "comments": context.iloc[position]["comments"],
        })
    return pd.DataFrame(records, columns=[
        "row", "patient_id", "side", "level", *sources, "raw_modalities",
        "hierarchy_changes", "disagreements", "comments",
    ])


def hierarchy_evidence(raw, disagreements, hierarchy):
    """Show raw parent/children and both reconstruction stages for disputed I/II."""
    records = []
    raw_sources = {name: observations(raw, name, ALL_LEVELS) for name in MODALITIES}
    for _, case in disagreements.iterrows():
        parent, side, index = case["level"], case["side"], case["row"]
        if parent not in SUBDIVISIONS:
            continue
        levels = [parent, *SUBDIVISIONS[parent]]

        def add(source, stage, table):
            values = [table.loc[index, (side, level)] for level in levels]
            if all(pd.isna(value) for value in values):
                return
            records.append({
                "patient_id": case["patient_id"], "location": f"{side} {parent}",
                "source": source, "stage": stage, "parent_value": values[0],
                "sublevels": ", ".join(f"{child} {_symbol(value)}"
                                       for child, value in zip(levels[1:], values[1:])),
            })

        for name, table in raw_sources.items():
            add("recorded_consensus" if name == "diagnostic_consensus" else name, "Recorded", table)
        for name, stages in hierarchy.items():
            add(name, "Before hierarchy", stages["before"])
            add(name, "After hierarchy", stages["after"])
    return pd.DataFrame(records, columns=[
        "patient_id", "location", "source", "stage", "parent_value", "sublevels",
    ])


def n_stage_comparison(raw, sources):
    """Screen N0/N+ against each source; list laterality without assigning TNM stages.

    Only N-stage prefixes establish its basis; the T-stage prefix is never used.
    Partial negative pathology is distinguished from complete negative coverage.
    """
    records = []
    context = patient_context(raw)
    for index, patient in context.iterrows():
        try:
            major, encoded_basis = _stage(patient["n_stage"])
            stage_valid = True
        except (ValueError, TypeError):
            major, encoded_basis, stage_valid = None, None, False
        prefix = "" if pd.isna(patient["n_stage_prefix"]) else str(patient["n_stage_prefix"]).strip().lower()
        prefix_basis = {"c": "clinical", "yc": "clinical", "p": "pathological", "yp": "pathological"}.get(prefix)
        if (prefix and prefix_basis is None) or (encoded_basis and prefix_basis and encoded_basis != prefix_basis):
            basis = "conflicting/invalid N prefix"
        else:
            basis = encoded_basis or prefix_basis or "unconfirmed"
        for name, table in sources.items():
            row = table.loc[index]
            known = int(row.notna().sum())
            positives = {side: [level for level in MAIN_LEVELS
                                if pd.notna(row[(side, level)]) and bool(row[(side, level)])]
                         for side in SIDES}
            any_positive = any(positives.values())
            if not stage_valid:
                status = "Invalid N-stage value"
            elif major is None:
                status = "N-stage unavailable"
            elif known == 0:
                status = "No involvement observations"
            elif major == 0 and any_positive:
                status = "Review: N0 with positive involvement"
            elif major > 0 and not any_positive:
                status = ("Review: N+ with all checked levels negative" if known == len(PAIR_COLUMNS)
                          else "Review: N+ without a positive; incomplete coverage")
            elif major == 0 and known < len(PAIR_COLUMNS):
                status = "N0 without observed positives; incomplete coverage"
            else:
                status = "Compatible at N0/N+ level"
            records.append({
                "patient_id": patient["patient_id"], "n_stage": patient["n_stage"],
                "stage_basis": basis, "source": name,
                "ipsi_positive_levels": ", ".join(positives["ipsi"]),
                "contra_positive_levels": ", ".join(positives["contra"]),
                "known_pairs": known, "assessment": status,
                "comments": patient["comments"],
            })
    return pd.DataFrame(records)


def _consensus_tables(raw):
    """Compare recorded, reconstructed and pathological labels on known pairs."""
    sources = {"recorded_consensus": observations(raw, "diagnostic_consensus")}
    sources.update({name: observations(raw, name)
                    for name in ["pathology", "CT", "MRI", "PET", "FNA", "pCT"]})
    raw_comparisons = [("recorded_consensus", name) for name in sources
                       if name != "recorded_consensus"]
    recorded_vs_modalities = comparison_summary(sources, raw_comparisons)

    reconstructed, modality_settings, reconstruction_hierarchy = reconstruct(raw)
    sources.update(reconstructed)
    without, with_pathology = "max_llh_without_pathology", "max_llh_with_pathology"
    reconstruction_comparisons = [
        ("recorded_consensus", without), ("recorded_consensus", with_pathology),
        (without, with_pathology),
    ]
    reconstruction_summary = comparison_summary(sources, reconstruction_comparisons)
    common_pathology_pairs = (
        sources["pathology"].notna() & sources["recorded_consensus"].notna()
        & sources[without].notna() & sources[with_pathology].notna()
    )
    pathology_comparisons = [(name, "pathology") for name in
                             ["recorded_consensus", without, with_pathology]]
    matched_pathology_summary = comparison_summary(sources, pathology_comparisons,
                                                   mask=common_pathology_pairs)
    reconstruction_changed = (
        sources[without].notna() & sources[with_pathology].notna()
        & sources[without].ne(sources[with_pathology]).fillna(False)
    )
    interpretation_summary = comparison_summary(
        sources, [("recorded_consensus", without), ("recorded_consensus", with_pathology)],
        mask=reconstruction_changed & common_pathology_pairs,
    )
    disagreements = disagreement_details(
        raw, sources, raw_comparisons + reconstruction_comparisons + pathology_comparisons,
        hierarchy=reconstruction_hierarchy,
    )
    # Ordinary disagreements with an individual modality provide context rather
    # than a reason to enter the main consensus/pathology review table.
    consensus_disagreements = disagreement_details(
        raw, sources, reconstruction_comparisons + pathology_comparisons,
        hierarchy=reconstruction_hierarchy,
    )
    sublevel_evidence = hierarchy_evidence(raw, disagreements, reconstruction_hierarchy)

    return sources, {
        "recorded_vs_modalities": recorded_vs_modalities,
        "modality_settings": modality_settings,
        "reconstruction_summary": reconstruction_summary,
        "matched_pathology_summary": matched_pathology_summary,
        "reconstruction_changed": reconstruction_changed,
        "interpretation_summary": interpretation_summary,
        "disagreements": consensus_disagreements,
        "all_disagreements": disagreements,
        "hierarchy_evidence": sublevel_evidence,
    }


def _stage_tables(raw, sources):
    """Summarize N-stage evidence and collect cases needing review."""
    without, with_pathology = "max_llh_without_pathology", "max_llh_with_pathology"
    stage_sources = {name: sources[name] for name in
                     ["recorded_consensus", without, with_pathology, "pathology"]}
    n_stage_details = n_stage_comparison(raw, stage_sources)
    # Preserve source row order, including missing IDs or repeated IDs from
    # different institutions. An ID alone is not a unique key for every centre.
    first_source = next(iter(stage_sources))
    n_stage_overview = n_stage_details.loc[
        n_stage_details["source"].eq(first_source), ["patient_id", "n_stage", "stage_basis"]
    ].reset_index(drop=True)

    def describe_involvement(row):
        if row["known_pairs"] == 0:
            return "No observations"
        if not row["ipsi_positive_levels"] and not row["contra_positive_levels"]:
            return f"No positives ({row['known_pairs']}/12 observed)"
        return (f"ipsi: {row['ipsi_positive_levels'] or 'none observed'}; "
                f"contra: {row['contra_positive_levels'] or 'none observed'}")

    for name in stage_sources:
        part = n_stage_details.loc[n_stage_details["source"].eq(name)].reset_index(drop=True)
        n_stage_overview[name] = part.apply(describe_involvement, axis=1)
    n_stage_review = n_stage_details.loc[
        n_stage_details["assessment"].str.startswith("Review:")
        | n_stage_details["assessment"].isin(["Invalid N-stage value", "N-stage unavailable"])
    ].reset_index(drop=True)

    return {
        "n_stage_details": n_stage_details,
        "n_stage_overview": n_stage_overview,
        "n_stage_review": n_stage_review,
    }
