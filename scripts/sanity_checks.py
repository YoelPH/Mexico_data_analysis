"""Additional checks for the unenhanced Mexico cohort; no input is modified.

Notebook usage:
    from scripts.sanity_checks import run_sanity_checks
    sanity_issues = run_sanity_checks(raw, mode="reduced")
    all_sanity_issues = run_sanity_checks(raw, mode="full")

When the basis of N staging is known, pass stage_basis="clinical" or
"pathological". Otherwise N-stage findings are requests for review. Explicit
c/p prefixes in a patient's N-stage field take precedence over that default.
Clinical staging is compared with diagnostic_consensus; pathological staging
with pathology and positive_dissected. With no known basis, all available
diagnostic modalities and positive counts are screened together.

This does not reproduce the notebook's consensus/pathology or hierarchy checks,
reconstruct an exact TNM category, or treat enhancement results as source data.
"""

import re
from datetime import date, datetime
from numbers import Real

import numpy as np
import pandas as pd


MODALITIES = ("diagnostic_consensus", "CT", "FNA", "MRI", "PET", "pCT", "pathology")
SIDES = ("ipsi", "contra")
MAIN_LEVELS = ("I", "II", "III", "IV", "V", "VII")
SUBDIVISIONS = {"I": ("Ia", "Ib"), "II": ("IIa", "IIb")}
SIDE_ES = {"ipsi": "ipsilateral", "contra": "contralateral"}
MODALITY_ES = {"diagnostic_consensus": "consenso diagnóstico", "pathology": "patología"}
MISSING = {"", "unknown", "not recorded", "nan", "none", "<na>", "na", "n/a"}
META = {"patient_id": ("patient", "core", "id"),
        "institution": ("patient", "core", "institution"),
        "subsite": ("tumor", "core", "subsite")}
ISSUE_COLUMNS = ["row", "patient_id", "institution", "subsite", "check", "category", "details"]
CONFLICT = "Posible contradicción"
INCOMPLETE = "Información incompleta"
REVIEW = "Revisar contexto"
INVALID = "Valor no válido"

# Full mode includes every historical check. Keep the reduced exclusions together
# so later changes to the reduced selection do not alter the full checks.
FULL_ONLY_CHECKS = frozenset({
    "invalid_date",
    "recurrence_before_diagnosis",
    "follow_up_before_diagnosis",
    "recurrence_after_follow_up",
    "n_stage_unavailable",
    "no_involvement_data",
    "side_without_involvement_data",
    "positive_exceeds_total",
    "positive_count_without_total",
    "negative_pathology_without_sample",
})


def _missing(value):
    return pd.isna(value) or (isinstance(value, str) and value.strip().casefold() in MISSING)


def _boolean(value):
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "1", "1.0"}:
            return True
        if text in {"false", "0", "0.0"}:
            return False
    raise ValueError("Expected true/false, 0/1, or missing")


def _number(value, integer=False):
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean in a numeric field")
    number = float(value)
    if not np.isfinite(number) or number < 0 or (integer and not number.is_integer()):
        raise ValueError("Expected a nonnegative finite number")
    return int(number) if integer else number


def _stage(value):
    """Return major category and optional c/p basis. NX/-1/missing have no category."""
    if _missing(value):
        return None, None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean N stage")
    text = re.sub(r"\s+", "", str(value)).lower()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = str(int(float(text)))
    if text == "-1":
        return None, None
    match = re.fullmatch(r"(yc|yp|c|p)?n?([0-3x])([abc]?)", text)
    if not match:
        raise ValueError("Unrecognized N stage")
    prefix, major, suffix = match.groups()
    if suffix and (major in {"0", "1", "x"} or (major == "3" and suffix == "c")):
        raise ValueError("Unrecognized N-stage suffix")
    basis = None if prefix is None else ("clinical" if prefix.endswith("c") else "pathological")
    return (None if major == "x" else int(major)), basis


def _identifier(value):
    if _missing(value):
        return None
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def _date(value, dayfirst):
    if _missing(value):
        return None
    # Workbook placeholders convey no date and do not participate in chronology.
    if isinstance(value, str) and value.strip().casefold() in {
        "not done", "no", "no les damos seguimiento", "not applicable",
    }:
        return None
    if isinstance(value, (bool, np.bool_)):
        if not value:
            return None
        raise ValueError("Boolean date")
    if isinstance(value, Real):
        # Unformatted Excel dates, not calendar years or YYYYMMDD integers.
        if not 10000 <= value <= 100000:
            raise ValueError("Unrecognized Excel date")
        result = pd.to_datetime(value, unit="D", origin="1899-12-30")
    elif isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        result = pd.Timestamp(value)
    elif re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", str(value).strip()):
        result = pd.Timestamp(value)
    else:
        result = pd.to_datetime(value, dayfirst=dayfirst, errors="raise")
    if pd.isna(result):
        raise ValueError("Unrecognized date")
    return result.tz_localize(None).normalize()


def run_sanity_checks(raw, *, mode="reduced", stage_basis=None,
                      main_levels=MAIN_LEVELS, subdivisions=None, dayfirst=True):
    """Return one row per issue, with Spanish explanations and notebook-compatible IDs.

    Pass the full unenhanced cohort, including patients without involvement data.
    Missing or invalid observations never become negative observations.
    mode="full" includes all historical checks. mode="reduced" (the default)
    omits date validity/chronology, missing N stage, missing involvement data,
    positive counts exceeding totals, positive counts without totals, and negative
    pathology with zero dissected nodes. See FULL_ONLY_CHECKS for the exact codes.
    In both modes, positive pathology with zero positive nodes is flagged only
    when the dissected total is known and greater than zero.
    Full-mode string dates are day-first by default; ISO and Excel dates are
    unambiguous. Reduced mode does not parse dates.
    Main levels define the coverage expected for the N-positive/all-negative check.
    Both sides are checked. Individual-node counts are never summed across levels.
    """
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 3:
        raise ValueError("Expected three-level columns: (modality, side, level).")
    if not raw.columns.is_unique:
        raise ValueError("Resolve duplicate column tuples before running the checks.")
    if mode not in ("full", "reduced"):
        raise ValueError("mode must be 'full' or 'reduced'.")
    excluded_checks = FULL_ONLY_CHECKS if mode == "reduced" else frozenset()
    if stage_basis not in {None, "clinical", "pathological"}:
        raise ValueError("stage_basis must be None, 'clinical', or 'pathological'.")
    subdivisions = SUBDIVISIONS if subdivisions is None else subdivisions
    main_levels = tuple(main_levels)
    if not main_levels:
        raise ValueError("main_levels must not be empty.")
    # Screen every anatomical level actually present, including any beyond I-V/VII.
    diagnostic_cols = [c for c in raw.columns if c[0] in MODALITIES and c[1] in SIDES]
    count_cols = [c for c in raw.columns
                  if c[0] in {"total_dissected", "positive_dissected"} and c[1] in SIDES]
    bool_cols = diagnostic_cols + [c for c in raw.columns if c[0] == "treated_LNLs" and c[1] in SIDES]
    for col in [("patient", "core", "neck_dissection"),
                ("tumor", "core", "central"), ("tumor", "core", "extension")]:
        if col in raw.columns:
            bool_cols.append(col)
    date_cols = ([c for c in raw.columns if c[2] in
                  {"date", "diagnose_date", "date_recurrence", "date_last_follow_up"}]
                 if mode == "full" else [])

    # Use centre + ID, never ID alone; do not group unidentified patients together.
    identities = [(_identifier(row.get(META["institution"])),
                   _identifier(row.get(META["patient_id"]))) for _, row in raw.iterrows()]
    identity_counts = {}
    for key in identities:
        if None not in key:
            identity_counts[key] = identity_counts.get(key, 0) + 1

    issues = []
    for position, (index, row) in enumerate(raw.iterrows()):
        info = {"row": index, **{name: row.get(col, pd.NA) for name, col in META.items()}}

        def add(code, category, text):
            if code not in excluded_checks:
                issues.append({**info, "check": code, "category": category, "details": text})

        key = identities[position]
        if key[0] is None or key[1] is None:
            add("missing_identity", INCOMPLETE,
                "Falta el centro o el ID del paciente. Completar la identificación antes de enviar la tabla.")
        elif identity_counts[key] > 1:
            add("duplicate_patient", REVIEW,
                f"El ID {key[1]} aparece en {identity_counts[key]} registros del mismo centro. "
                "Confirmar si son duplicados o episodios distintos.")

        booleans = {}
        for col in bool_cols:
            try:
                booleans[col] = _boolean(row[col])
            except (ValueError, TypeError):
                booleans[col] = None
                add("invalid_boolean", INVALID,
                    f"{' / '.join(col)}: valor {row[col]!r}; se esperaba True, False o información no registrada.")

        counts = {}
        for col in count_cols:
            try:
                counts[col] = _number(row[col], integer=True)
            except (ValueError, TypeError, OverflowError):
                counts[col] = None
                add("invalid_node_count", INVALID,
                    f"{' / '.join(col)}: recuento {row[col]!r}; debe ser un entero no negativo.")

        for col, label in [(("patient", "core", "age"), "Edad"),
                           (("tumor", "core", "volume"), "Volumen tumoral")]:
            if col in raw.columns:
                try:
                    _number(row[col])
                except (ValueError, TypeError, OverflowError):
                    add("invalid_numeric", INVALID,
                        f"{label}: valor {row[col]!r}; se esperaba un número no negativo.")

        observed = [c for c in diagnostic_cols if booleans[c] is not None]
        if not observed:
            add("no_involvement_data", INCOMPLETE,
                "No hay datos válidos de afectación ganglionar en ninguna modalidad ni lado. "
                "Registrar positivos, negativos o información desconocida según corresponda.")
        else:
            for side in SIDES:
                if not any(c[1] == side for c in observed):
                    add("side_without_involvement_data", INCOMPLETE,
                        f"No hay datos válidos de afectación del lado {SIDE_ES[side]} en ninguna modalidad.")

        stage_value = row.get(("patient", "core", "n_stage"))
        valid_stage = True
        try:
            n_stage, encoded_basis = _stage(stage_value)
        except (ValueError, TypeError):
            valid_stage = False
            n_stage, encoded_basis = None, None
            add("invalid_n_stage", INVALID,
                f"Estadio N no reconocido: {stage_value!r}. Verificar la categoría registrada.")

        prefix_value = row.get(("patient", "core", "n_stage_prefix"))
        prefix = "" if _missing(prefix_value) else str(prefix_value).strip().lower()
        prefix_basis = {"c": "clinical", "yc": "clinical", "p": "pathological", "yp": "pathological"}.get(prefix)
        if prefix and prefix_basis is None:
            add("invalid_n_stage_prefix", INVALID,
                f"Prefijo del estadio N no reconocido: {prefix_value!r}. Confirmar si es clínico o patológico.")
        basis_conflict = encoded_basis and prefix_basis and encoded_basis != prefix_basis
        if basis_conflict:
            add("n_stage_prefix_conflict", CONFLICT,
                "El prefijo incluido en el estadio N contradice n_stage_prefix. Confirmar el origen del estadio.")
        basis = None if basis_conflict else (encoded_basis or prefix_basis or stage_basis)

        if valid_stage and n_stage is None:
            add("n_stage_unavailable", INCOMPLETE,
                f"Estadio N {'no registrado' if _missing(stage_value) else str(stage_value)}: "
                "no se puede comprobar su concordancia con la afectación ganglionar.")
        elif valid_stage:
            selected = (("diagnostic_consensus",) if basis == "clinical" else
                        ("pathology",) if basis == "pathological" else MODALITIES)
            source = ("consenso diagnóstico" if basis == "clinical" else
                      "patología y recuentos positivos" if basis == "pathological" else
                      "todas las modalidades y recuentos positivos disponibles")
            positives = [f"{MODALITY_ES.get(c[0], c[0])}, {SIDE_ES[c[1]]}, {c[2]}"
                         for c in diagnostic_cols if c[0] in selected and booleans[c] is True]
            if basis != "clinical":
                positives += [f"ganglios positivos disecados, {SIDE_ES[c[1]]}, {c[2]}: {n}"
                              for c, n in counts.items()
                              if c[0] == "positive_dissected" and n is not None and n > 0]
            context = (" Confirmar si el estadio es clínico o patológico y que los datos "
                       "correspondan a la misma evaluación." if basis is None else
                       " Confirmar que estadio y observaciones correspondan a la misma evaluación.")

            if n_stage == 0 and positives:
                add("n0_with_positive_involvement", REVIEW if basis is None else CONFLICT,
                    f"N0 con afectación positiva en {source}: {'; '.join(positives)}." + context)
            elif n_stage > 0 and not positives:
                uncovered = []
                for side in SIDES:
                    for level in main_levels:
                        # A negative parent suffices; otherwise ALL children must be negative
                        # in the SAME modality. A lone negative child is insufficient.
                        children = subdivisions.get(level, ())
                        negative = any(
                            booleans.get((mod, side, level)) is False or
                            (bool(children) and all(booleans.get((mod, side, sub)) is False for sub in children))
                            for mod in selected
                        )
                        if not negative:
                            uncovered.append(f"{SIDE_ES[side]} {level}")
                if uncovered:
                    add("n_positive_without_localization", INCOMPLETE,
                        f"N{n_stage} sin afectación positiva registrada en {source}. "
                        f"Regiones sin información negativa suficiente: {', '.join(uncovered)}. "
                        "Indicar dónde se documentó la afectación ganglionar." + context)
                else:
                    add("n_positive_all_regions_negative", REVIEW if basis is None else CONFLICT,
                        f"N{n_stage} sin afectación positiva en {source}, con observaciones negativas "
                        "para todas las regiones comprobadas en ambos lados. Verificar el estadio, "
                        "los valores negativos y si la afectación estaba fuera de estas regiones." + context)

        # Compare matching side/level counts only: no parent/child summation.
        unsampled_negative = {side: [] for side in SIDES}
        for side, level in sorted({c[1:] for c in count_cols}):
            total = counts.get(("total_dissected", side, level))
            positive = counts.get(("positive_dissected", side, level))
            pathology = booleans.get(("pathology", side, level))
            location = f"lado {SIDE_ES[side]}, {'total del lado' if level == 'all' else 'nivel ' + level}"
            if total is not None and positive is not None and positive > total:
                add("positive_exceeds_total", CONFLICT,
                    f"{location}: {positive} ganglios positivos, pero solo {total} disecados. "
                    "Verificar que los recuentos correspondan a la misma muestra.")
            if positive is not None and positive > 0:
                if total is None:
                    add("positive_count_without_total", INCOMPLETE,
                        f"{location}: {positive} ganglios positivos sin un total disecado válido.")
                if pathology is False:
                    add("positive_count_negative_pathology", CONFLICT,
                        f"{location}: patología negativa, pero {positive} ganglios positivos. "
                        "Verificar que ambos datos correspondan a la misma muestra y evaluación.")
            # Require a known nonzero sample and an explicitly recorded zero positives.
            # Zero/unknown sampling and unknown positive counts do not establish a conflict.
            if total is not None and total > 0 and positive == 0 and pathology is True:
                add("zero_positive_count_positive_pathology", CONFLICT,
                    f"{location}: patología positiva, pero 0 positivos de {total} ganglios disecados. "
                    "Verificar que los datos correspondan a la misma muestra y evaluación.")
            if total == 0 and pathology is False:
                unsampled_negative[side].append(level)
        if any(unsampled_negative.values()):
            locations = "; ".join(f"{SIDE_ES[side]}: {', '.join(levels)}"
                                  for side, levels in unsampled_negative.items() if levels)
            add("negative_pathology_without_sample", REVIEW,
                f"Patología negativa con 0 ganglios disecados en {locations}. Confirmar si existen "
                "otras muestras que justifiquen estos resultados o si deben registrarse como desconocidos.")

        neck_dissection = booleans.get(("patient", "core", "neck_dissection"))
        any_dissected = any(c[0] == "total_dissected" and n is not None and n > 0 for c, n in counts.items())
        any_count = any(n is not None for n in counts.values())
        any_pathology = any(c[0] == "pathology" and booleans[c] is not None for c in diagnostic_cols)
        if neck_dissection is False and any_dissected:
            add("dissection_flag_vs_counts", CONFLICT,
                "Disección cervical = False, pero se registran ganglios disecados. Verificar cirugía y recuentos.")
        if neck_dissection is True and not any_count and not any_pathology:
            add("dissection_without_results", INCOMPLETE,
                "Disección cervical = True, sin datos válidos de patología ni recuentos ganglionares.")

        dates = {}
        for col in date_cols:
            try:
                dates[col] = _date(row[col], dayfirst)
            except (ValueError, TypeError, OverflowError):
                dates[col] = None
                add("invalid_date", INVALID, f"{' / '.join(col)}: fecha no válida {row[col]!r}.")
        diagnosis = dates.get(("patient", "core", "diagnose_date"))
        recurrence = dates.get(("outcome", "core", "date_recurrence"))
        follow_up = dates.get(("outcome", "core", "date_last_follow_up"))
        for code, earlier, later, explanation in [
            ("recurrence_before_diagnosis", recurrence, diagnosis, "La recurrencia precede al diagnóstico inicial"),
            ("follow_up_before_diagnosis", follow_up, diagnosis, "El último seguimiento precede al diagnóstico inicial"),
            ("recurrence_after_follow_up", follow_up, recurrence, "La recurrencia es posterior al último seguimiento registrado"),
        ]:
            if earlier is not None and later is not None and earlier < later:
                add(code, REVIEW, explanation + ". Verificar las fechas y el episodio al que corresponden.")

    result = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    result.attrs["missing_columns"] = [c for c in [*META.values(), ("patient", "core", "n_stage")]
                                       if c not in raw.columns]
    result.attrs["stage_basis_default"] = stage_basis
    result.attrs["expected_main_levels"] = main_levels
    result.attrs["mode"] = mode
    return result
