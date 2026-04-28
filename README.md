# HMDA Responsible ML Capstone

This project contains a Jupyter notebook for a Responsible ML capstone built on the HMDA 2024 loan-level dataset.

## Project Scope

The notebook currently covers both the baseline modeling foundation and the first fairness-analysis extension:

- raw HMDA data inspection with DuckDB
- filtering and parquet export
- binary label creation from `action_taken`
- baseline feature selection
- data cleaning and preprocessing
- logistic regression baseline model
- XGBoost main model
- random forest second comparison model
- model evaluation and model-comparison tables
- Lecture 03 subgroup fairness diagnostics
- intersectional subgroup analysis

This notebook now serves as the working capstone notebook for baseline modeling plus the first round of fairness diagnostics. Later phases can continue to extend it with explainability, robustness, and reporting work.

## Files

- `hmda_responsible_ml_capstone.ipynb`: main notebook deliverable
- `pyproject.toml` and `uv.lock`: environment managed with `uv`
- `2024_lar.zip`: raw HMDA archive placed in the project root manually
- `2024_lar.txt`: extracted raw HMDA text file created after the first full run
- `data/hmda_filtered.parquet`: filtered modeling dataset created by the notebook
- `data/hmda_filtered_preview.csv`: small preview export created by the notebook
- `assignment_ready_explanation.md`: English write-up of the modeling and feature-selection choices

## Environment

The project now uses a Python 3.12-compatible environment so that the full capstone dependency set can coexist, including `solas-ai`.

- required Python range: `>=3.12,<3.13`
- dependency management: `uv`
- environment reproduction: `uv sync`

The environment has been validated with:

- baseline notebook execution
- `duckdb`, `scikit-learn`, `shap`, `lime`, `dice-ml`, `statsmodels`, `xgboost`
- `solas-ai` package installation

Note:

- `solas-ai` installs successfully in this environment, but the package exposes `solas_disparity` as the available importable module in the current setup.
- `xgboost` required the system OpenMP runtime on macOS. `libomp` has already been installed on this machine so `xgboost` can load correctly.

## Install and Run

1. Sync the environment:

   ```bash
   uv sync
   ```

2. Open `hmda_responsible_ml_capstone.ipynb`.

## Data Requirement

Before running the notebook, make sure that:

- the raw HMDA archive `2024_lar.zip` has already been downloaded manually
- the file is placed in the project root directory
- the notebook can access local project files

The notebook does not include automatic download logic. It assumes the HMDA archive already exists locally.

On the first full run, the notebook extracts `2024_lar.txt` from `2024_lar.zip` if needed, then uses DuckDB to filter the raw data and save `data/hmda_filtered.parquet` for reuse.


## Modeling Goal

The current notebook builds a baseline binary classifier for HMDA 2024 loan applications.

- `action_taken` values `1` and `2` are mapped to `label = 1` for approved outcomes
- `action_taken` value `3` is mapped to `label = 0` for denied outcomes
- all other `action_taken` values are excluded from the modeling dataset

This means the current goal is to predict approved versus denied applications, not every possible HMDA process outcome.

This baseline models the lender's approval-versus-denial decision, not downstream loan performance, default risk, or final loan origination completion.

## Responsible ML Framing

The notebook opening frames the project as a loan approval decision-support problem with both predictive and social consequences.

- stakeholders include applicants, financial institutions, and regulators
- the notebook explicitly treats this as more than an accuracy-only task
- the modeling objective is to balance predictive performance with subgroup fairness diagnostics
- failure is not defined only as low aggregate performance, but also as systematic errors or adverse impact across demographic groups

## Why Only `action_taken` 1, 2, and 3

The notebook keeps only `action_taken in (1, 2, 3)` because those values align cleanly with the binary prediction target.

- `1`: originated
- `2`: approved but not accepted
- `3`: denied

Other values are excluded because they describe different process states rather than the same final underwriting decision.

- `4`: withdrawn by applicant
- `5`: file closed for incompleteness
- `6`: purchased loan
- `7`: preapproval denied
- `8`: preapproval approved but not accepted

Keeping those rows would mix different business processes into one label and make the baseline harder to interpret.

## Data Cleaning and Preprocessing

The notebook uses DuckDB first because the raw HMDA text file is very large.

1. Inspect the raw schema directly from the pipe-delimited file.
2. Select a smaller set of columns needed for the baseline phase.
3. Filter rows to `action_taken in ('1', '2', '3')`.
4. Export the filtered result to `data/hmda_filtered.parquet`.
5. Load the parquet file into pandas for modeling.
6. Restrict the modeling dataframe to the required columns.
7. Create the binary target from `action_taken`.
8. Convert numeric-looking fields to numeric types with `pd.to_numeric(..., errors="coerce")`.
9. Treat common missing tokens such as `NA`, `Exempt`, and empty strings as missing values.
10. Impute numeric features with the median and categorical features with the most frequent value.
11. One-hot encode categorical variables and standardize numeric variables.

## Selected Baseline Features

The current notebook uses 13 model features.

Numeric features:

- `property_value`
- `income`
- `tract_population`
- `tract_minority_population_percent`
- `ffiec_msa_md_median_family_income`
- `tract_to_msa_income_percentage`

Categorical features:

- `state_code`
- `derived_loan_product_type`
- `derived_dwelling_category`
- `loan_purpose`
- `lien_status`
- `occupancy_type`
- `applicant_age`

Protected-group columns retained for subgroup diagnostics but not used for baseline model training:

- `derived_race`
- `derived_sex`
- `derived_ethnicity`

During DuckDB filtering, the notebook keeps only rows where `derived_sex` is `Male` or `Female`. This means the filtered dataset size reported in the notebook matches the rows that are actually carried into modeling and sex-based subgroup evaluation.

## Why These Features Were Used

These features were chosen because they provide a simple and defensible baseline.

- property value and income provide high-level financial context without relying on the disallowed loan-amount or loan-structure fields
- product, dwelling, purpose, lien, occupancy, and age capture major application characteristics
- tract-level variables add neighborhood socioeconomic context
- state helps capture broad geographic differences without using highly granular IDs
- `interest_rate`, `loan_amount`, `combined_loan_to_value_ratio`, and `loan_term` are intentionally excluded from the model inputs in the current notebook revision

## Why Many Other HMDA Features Were Not Used Yet

The raw file contains 99 columns, but many were intentionally excluded from the first baseline.

- Identifier or very high-cardinality fields such as `lei`, `county_code`, `census_tract`, and `derived_msa_md` were excluded to avoid sparse, overly specific representations.
- Post-outcome or leakage-prone fields such as `denial_reason_1` to `denial_reason_4` and `purchaser_type` were excluded because they reveal information too close to the outcome itself.
- Detailed applicant and co-applicant demographic fields were excluded to keep the first baseline simple and easier to preprocess.
- Sensitive attributes such as race, sex, and ethnicity were not used as training features in the baseline model, but they were retained for later subgroup diagnostics and future fairness analysis.
- Specialized product fields, such as several manufactured-home and non-amortizing-payment fields, were deferred to keep the first model compact and interpretable.

## Current Model Setup

The notebook now uses:

- `LogisticRegression(max_iter=1000)` as the baseline model
- `XGBClassifier(random_state=42)` as the main model
- `RandomForestClassifier(random_state=42)` as a lighter second comparison model
- no naive baseline block
- no `interest_rate` feature in the modeling inputs

The evaluation tables now report `accuracy`, `AUC`, `log-loss`, `precision`, and `recall`, along with a compact `FPR` summary and a three-model comparison table.

## Lecture 03 Fairness Extension

After the model-comparison section, the notebook now includes a Lecture 03 fairness-analysis block built around the XGBoost main model.

That section adds:

- AIR analysis by `derived_race` with `White` as the reference group
- AIR analysis by `derived_sex` with `Male` as the reference group
- AIR analysis by `derived_ethnicity` with `Not Hispanic or Latino` as the reference group
- subgroup error-rate tables by race, sex, and ethnicity
- intersectional subgroup analysis for `race × sex`

The AIR sections use:

- a manual subgroup-rate calculation
- `solas_disparity` for pairwise adverse impact ratio summaries
- the 4/5ths rule to flag groups with AIR below `0.80`

The subgroup error-rate section reports:

- `Accuracy`
- `FPR`
- `FNR`

for each protected-group breakdown on the held-out test set.

The intersectional section constructs combined `race / sex` subgroup labels, filters out very small groups using `min_n=30`, and identifies the worst-performing subgroup relative to `White / Male`.

## Current Notebook Status

The notebook is now broader than a simple baseline:

- it builds the filtered HMDA modeling dataset
- trains and compares three models
- evaluates them with compact performance tables
- carries protected-group columns forward for fairness analysis
- runs Lecture 03 subgroup and intersectional diagnostics on the XGBoost predictions

This notebook is still not positioned as a decision-ready system. It should be treated as a capstone workflow that supports later explainability, robustness, and deeper responsible-ML analysis. If you want refreshed full-dataset metrics or fairness outputs after future edits, rerun the notebook from top to bottom.
