# Data checks

Start with a dataset notebook: `abc_check.ipynb`, `inc_check.ipynb`,
`mid_check.ipynb`, `ms_check.ipynb`, `nov_check.ipynb`, `ueo_check.ipynb`,
or `xxi_check.ipynb`. Each selects its workbook and calls the same analysis.

## Code map

| File | What to read there |
| --- | --- |
| `scripts/data_checks.py` | Start at `analyse_dataset()`. It runs the four notebook analyses: pathology availability, missingness, consistency, and consensus/N-stage comparisons. |
| `scripts/sanity_checks.py` | The individual rules for identity, invalid values, N-stage, node counts and dates. |
| `scripts/notebook_display.py` | Table labels, formatting and expandable evidence. It does not decide involvement. |
| `tests/` | Automated checks, separate from the analysis code. |

Both reconstructed consensuses exclude the recorded consensus. One uses
CT/MRI/PET/pCT/FNA; the other also uses pathology. Parent/sublevel reconciliation
is included, with values before and after available in the evidence tables.
Missing values remain unknown, and the source data is never changed.

The dataset notebooks use the reduced sanity checks. `mode="full"` additionally
checks dates, missing coverage/staging and the extra node-count rules listed in
`FULL_ONLY_CHECKS`.

The older pooled notebooks are kept for reference. They are not needed to run
the individual dataset checks.

## Run the tests

From `Mexico_data_analysis/`:

```bash
../.venv/bin/python -B -m unittest discover -s tests
```
