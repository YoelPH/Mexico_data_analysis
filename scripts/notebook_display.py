"""Notebook presentation helpers; the analysis tables are never modified here."""

from html import escape

import pandas as pd
from IPython.display import HTML, display


SOURCE_LABELS = {
    "recorded_consensus": "Recorded consensus",
    "max_llh_without_pathology": "Without pathology",
    "max_llh_with_pathology": "With pathology",
    "pathology": "Pathology",
}

_STYLE = """
<style>
.data-check-report { color: inherit; margin: 0.5em 0 1.2em; line-height: 1.5; }
.data-check-report .table-scroll { overflow-x: auto; }
.data-check-report table { border-collapse: collapse; font-size: 0.94em; width: auto; }
.data-check-report th, .data-check-report td {
    padding: 0.55em 0.8em; text-align: left; vertical-align: top;
    white-space: normal; max-width: 30em; border-bottom: 1px solid rgba(128,128,128,.22);
}
.data-check-report th { font-weight: 600; background: rgba(128,128,128,.08); }
.data-check-report summary { cursor: pointer; padding: 0.35em 0; font-weight: 500; }
.data-check-report details[open] summary { margin-bottom: 0.5em; }
.data-check-report .takeaway {
    border-left: 3px solid #005ea8; padding: 0.55em 0.9em;
    background: rgba(0,94,168,.06);
}
</style>
"""


def show_note(title, text=""):
    body = f"<strong>{escape(str(title))}</strong>"
    if text:
        body += f"<br>{escape(str(text))}"
    display(HTML(_STYLE + f'<div class="data-check-report"><div class="takeaway">{body}</div></div>'))


def show_table(frame, *, details=None, binary=(), percentages=()):
    """Render wrapped, untruncated text; optional details are closed by default."""
    def diagnosis(value):
        return "—" if pd.isna(value) else "+" if bool(value) else "−"

    formatters = {column: diagnosis for column in binary}
    formatters.update({column: lambda value: f"{value:.1f}%" for column in percentages})
    with pd.option_context("display.max_colwidth", None):
        table = frame.to_html(index=False, na_rep="—", escape=True, border=0,
                              formatters=formatters, float_format=lambda value: f"{value:g}")
    body = f'<div class="table-scroll">{table}</div>'
    if details:
        body = f"<details><summary>{escape(details)}</summary>{body}</details>"
    display(HTML(_STYLE + f'<div class="data-check-report">{body}</div>'))


def comparison_view(summary, labels=None):
    """Keep the headline comparison to five short columns."""
    view = summary[["comparison", "compared_pairs", "disagreements",
                    "patients_affected", "agreement_pct"]].copy()
    if labels is not None:
        view["comparison"] = labels
    return view.rename(columns={
        "comparison": "Comparison", "compared_pairs": "Known pairs",
        "disagreements": "Disagreements", "patients_affected": "Patients affected",
        "agreement_pct": "Agreement",
    })
