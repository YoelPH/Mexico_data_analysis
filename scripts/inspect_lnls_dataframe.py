"""Recreate and inspect the cohort using the saved Data_check.ipynb cells."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lydata  # Registers the pandas .ly accessor.


def load_training_dataframe(allow_partial=False):
    notebook = json.loads(Path("Data_check.ipynb").read_text(encoding="utf-8"))
    namespace = {"pd": pd, "np": np}
    for line in notebook["cells"][4]["source"]:
        try:
            exec(line, namespace)
        except PermissionError:
            if not allow_partial:
                raise
            name = line.split("=", 1)[0].strip()
            namespace[name] = pd.DataFrame()
            print("PARTIAL INSPECTION: locked input omitted:", line.strip())
    for cell_number in (5, 6, 7, 8):
        exec("".join(notebook["cells"][cell_number]["source"]), namespace)
    return namespace["datasets"], namespace["dataset_filled"]


def inspect_dataframe(data, partial=False):
    print("Dataframe type:", type(data))
    print("Shape:", data.shape)
    print("Column index type:", type(data.columns))
    print("Column levels:", data.columns.nlevels, data.columns.names)
    print("Row index:", type(data.index), "unique:", data.index.is_unique)
    print("Duplicate columns:", data.columns.duplicated().sum())
    print("Dtype counts:", data.dtypes.astype(str).value_counts().to_dict())
    print("Top-level headers:", data.columns.get_level_values(0).unique().tolist())
    print("Second-level headers:", data.columns.get_level_values(1).unique().tolist())
    print("Subsites:", data[("tumor", "core", "subsite")].value_counts(dropna=False).to_dict())
    id_columns = [("patient", "core", name) for name in ("institution", "id")]
    print("Duplicate institution/patient pairs:", data.duplicated(id_columns, keep=False).sum())
    print("Missing institution/ID:", data[id_columns].isna().sum().to_dict())
    records = []
    headers = {"max_llh", "pathology", "CT", "FNA", "MRI", "PET", "pCT", "diagnostic_consensus"}
    for column in data.columns:
        if column[0] not in headers or column[1] not in {"ipsi", "contra"}:
            continue
        series = data[column]
        records.append({
            "modality": column[0], "side": column[1], "LNL": column[2],
            "dtype": str(series.dtype), "nonmissing": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "value_types": ", ".join(sorted({type(v).__name__ for v in series.dropna()})),
            "values": repr(series.value_counts(dropna=False).to_dict()),
        })
    audit = pd.DataFrame(records)
    print(audit.to_string(index=False))
    output = Path("bilateral_pathology_verification")
    output.mkdir(exist_ok=True)
    filename = "partial_dataframe_audit.csv" if partial else "source_dataframe_audit.csv"
    audit.to_csv(output / filename, index=False)
    for header in ("total_dissected", "positive_dissected", "enbloc_dissected"):
        print(header, [c for c in data.columns if c[0] == header])


if __name__ == "__main__":
    partial = "--available-only" in sys.argv
    raw, filled = load_training_dataframe(allow_partial=partial)
    print("Before enhancement:", raw.shape)
    inspect_dataframe(filled, partial=partial)
