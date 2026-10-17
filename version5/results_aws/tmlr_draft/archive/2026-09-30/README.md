# Mixed-activation article draft

**Title:** When Mixed Activations Help Neural Networks Resist Training Collapse  
**Authors:** Barkev Şarklı; Rahim Dehkharghani  
**Version:** Advisor-review draft, 30 September 2026

The manuscript has 24 pages, 23 numbered tables, and 14 numbered figures, including the appendices. The writing uses short, plain English sentences while retaining terms needed to report the methods. It covers all six completed experiment groups, not only results that favor mixing.

## Read and edit

- `main.pdf`: compiled, named draft.
- `main.tex`: main manuscript source.
- `tables/`: numerical tables included by the main source.
- `figures/`: vector PDF figures used by the manuscript.
- `references.bib`: bibliography.
- `tmlr.sty`, `tmlr.bst`, `TMLR_STYLE_LICENSE`: official TMLR style and its license.
- `analysis/`: analysis scripts, derived CSVs, input hashes, and verification report.
- `advisor_notes_tr.md`: short Turkish notes on the main claims and open issues.

The ZIP contains everything needed to compile the manuscript; it does not require the original logs for LaTeX compilation. Upload the ZIP to Overleaf and choose `main.tex`, or run this command inside the extracted folder with a TeX installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript uses the official named **preprint** option. This is the requested advisor copy, not an anonymous review submission. For a blind copy, change `\usepackage[preprint]{tmlr}` to `\usepackage{tmlr}`, remove the advisor/date note, clear author names from PDF metadata, and check any identifying supplementary material. Do not change the style dimensions.

Barkev's supplied email is included, together with “Undergraduate student.” Institutional affiliations and Rahim's email are omitted because they were not supplied. Author approval and final affiliations remain to be completed before submission.

## Reproduce the analysis

Requirements: Python with numpy, pandas, scipy, and matplotlib. The original 32 CSV files are in the experiment results directory, outside this ZIP. Their filenames, row counts, and SHA-256 hashes are recorded in `analysis/verification.json`. The original training source is also outside this package.

Set the location of those CSVs and run:

```sh
export EXPERIMENT_RESULTS_DIR='/absolute/path/to/results_aws'
python3 analysis/analyze_results.py
python3 analysis/build_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If the package remains inside the original `results_aws` directory, the scripts use that parent directory by default. The prior `research_audit/paired_comparisons.csv` file is optional; when present, its mean differences are checked against the new calculations.

The analysis removes repeated header lines, checks final and checkpoint confusion-matrix metrics, checks validation histories and selected epochs, and keeps repeated jobs in different groups from being counted as extra seeds. It uses bootstrap seed 20260915 and 10,000 paired resamples. It makes no new training runs.

## Scope of the evidence

The main claim concerns late training collapse and learning-rate sensitivity under the tested setup. It does not establish resistance to adversarial attacks or noisy inputs. It does not establish a general accuracy or speed advantage. CIFAR-10 counterexamples, pure sigmoid's strong fixed-rate scores, tuning-rule sensitivity, the MSE derivative scaling issue, and missing follow-up runs remain visible in the text.

The official format was obtained from https://github.com/JmlrOrg/tmlr-style-file and checked against https://jmlr.org/tmlr/author-guide.html on 30 September 2026. The style file is unmodified. Scientific content, claims, and readiness for submission should be reviewed by both authors.
