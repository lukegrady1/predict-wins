# claude.md — Predict Wins: Fix R Pipeline, Align Train/Predict, and Stop Constant Probabilities

## Context
This repo builds NFL win probabilities. Current symptoms:
- Upcoming script drops `home_team` / `away_team` too early, then tries to use them later → `object 'away_team' not found`.
- Train-time and predict-time matrices don’t match (different columns / order / dummy levels) → XGBoost outputs near-constant probs (~0.997 for all games).
- Recipe/preprocessing drift: dummying, imputation, zero-variance removal at train-time aren’t reused at predict-time.

## Goal
Make the model run end-to-end and produce **sane, non-constant** probabilities by:
1) Reusing the exact **preprocessing recipe** (dummy levels, imputation, zero-variance) at **predict-time**,  
2) Enforcing **identical feature set and column order** as training,  
3) Creating `matchup` from `away_team` / `home_team` **before** any step that drops those columns,  
4) Adding guardrails/tests so this never silently regresses.

---

## What to examine (scan everything)
Search the entire repo (both `scripts/` and `R/`) for **all .R files**. Specifically audit:
- `scripts/02_build_features.R`
- `R/features_build.R` and helpers (`R/utils_io.R`, etc.)
- `scripts/03_train_xgb.R` (create/update)
- `scripts/06_predict_upcoming.R`
- Any other R scripts that: 
  - make dummies, impute, select columns, or call xgboost,
  - write/read parquet/csv that feed train/predict,
  - manipulate `home_team`/`away_team`/`matchup`.

Make all changes via a single PR with atomic commits per file so diffs are readable.

---

## Required changes (authoritative plan)

### A) Train-time: save the exact preprocessing + feature order + model
- Build a **recipes** pipeline on `train_raw` **before manual dummies**:
  - `update_role(game_id, season, week, new_role = "id")`
  - `step_rm(game_id, season, week)`
  - `step_impute_median(all_numeric_predictors())`
  - `step_dummy(all_nominal_predictors(), one_hot = TRUE)`
  - `step_zv(all_predictors())`
- `prep()` the recipe and **bake** training data to `trainX` (drop `home_win`).
- Persist artifacts:
  - `models/rec_prepped.rds` (the prepped recipe)
  - `models/xgb_feature_names.rds` (exact `colnames(trainX)` order)
  - `models/winprob_xgb.model` (trained xgboost model)
- Train XGBoost on `data.matrix(trainX)` and save with `xgboost::xgb.save`.

Create/replace `scripts/03_train_xgb.R` implementing the above.

### B) Predict-time: create `matchup` first, then bake with the saved recipe, align columns
- In `scripts/06_predict_upcoming.R`:
  - **Load** `rec_prepped.rds`, `xgb_feature_names.rds`, and model.
  - Build `upcoming_raw` that **includes** `home_team` and `away_team`.
  - **Create** `matchup = paste0(away_team, " @ ", home_team)` **before** any select/remove that drops team strings.
  - `bake(rec, new_data = upcoming_raw)` to get `upcoming_baked`.
  - Add any **missing** columns from `x_cols` with zeros; drop any **extra** columns not in `x_cols`; then **reorder** to `x_cols`. `stopifnot(identical(colnames(upcoming_baked), x_cols))`.
  - Predict with `predict(bst, data.matrix(upcoming_baked))` and bind back to identifiers for output parquet/csv.

### C) Guardrails to block silent regressions
Add to `scripts/06_predict_upcoming.R` **before** prediction:
```r
nz <- vapply(upcoming_baked, function(x) length(unique(x)), integer(1))
if (mean(nz <= 1) > 0.3) {
  stop("Too many near-constant columns in upcoming features; check joins and recipe.")
}
```
Also warn if the model objective isn’t `binary:logistic`.

### D) Feature-build hygiene (`R/features_build.R` et al.)
- Turnovers from PBP: define `is_turnover` as `interception == 1 | fumble_lost == 1` (don’t rely on a non-existent `turnover` column).
- Third-down attempt: `is_third_down_att <- (down == 3L) & (is_pass | is_rush)`; conversions use `third_down_converted` AND the above.
- Never hand-roll dummies in build scripts; rely on the **recipe** only.
- Ensure `02_build_features.R` writes a clean `data_processed/train_raw.parquet` with `home_win`, ids, and raw factors/numerics (no manual dummying).

---

## Exact file edits to apply

### 1) `scripts/02_build_features.R`
- Ensure it **outputs** `data_processed/train_raw.parquet` (contains `home_win`, `game_id`, `season`, `week`, and raw features). Do **not** drop team strings here if they’re needed elsewhere downstream.
- Do **not** do any dummy-coding here; let the recipe handle that.
- Confirm columns exist: `stopifnot(all(c("home_win","game_id","season","week") %in% names(train_raw)))` after you build it.

### 2) `R/features_build.R`
Inside your `summarize_team_game()` (or equivalent):
```r
is_pass <- dplyr::coalesce(.data$pass == 1, FALSE)
is_rush <- dplyr::coalesce(.data$rush == 1, FALSE)

is_third_down_att  <- (.data$down == 3L) & (is_pass | is_rush)
is_third_down_conv <- dplyr::coalesce(.data$third_down_converted, FALSE) & is_third_down_att

is_turnover <- dplyr::coalesce(.data$interception == 1, FALSE) |
               dplyr::coalesce(.data$fumble_lost  == 1, FALSE)
```
Remove any references to a single `turnover` column if it doesn’t exist.

### 3) `scripts/03_train_xgb.R` (add new)
Implement the full **train-time** recipe → bake → save artifacts → train XGB → save model and `x_cols`. Use conservative defaults for XGB (tunable later).

### 4) `scripts/06_predict_upcoming.R` (replace)
- Load artifacts (`rec`, `x_cols`, `bst`).
- Read `data_processed/upcoming_raw.parquet` (or build it) **with** `home_team/away_team` intact.
- `mutate(matchup = paste0(away_team, " @ ", home_team))` **before** baking.
- `upcoming_baked <- bake(rec, new_data = upcoming_raw)`.
- Align to `x_cols` (add missing zeros, drop extras, reorder); `stopifnot(identical(colnames(upcoming_baked), x_cols))`.
- Guardrails on near-constant columns.
- Predict and write `data/upcoming_predictions_wk{params$week}.(parquet|csv)`.

---

## Tests & sanity checks (must pass)

1) After training:
```r
length(readRDS("models/xgb_feature_names.rds")) > 0
xgboost::xgb.attributes(xgboost::xgb.load("models/winprob_xgb.model"))$objective == "binary:logistic"
```
2) Before predicting:
```r
head(upcoming_raw, 3) %>% select(season, week, game_id, home_team, away_team)
identical(colnames(bake(readRDS("models/rec_prepped.rds"), new_data = upcoming_raw)) %>% 
  { intersect(., readRDS("models/xgb_feature_names.rds")) } %>% length() > 0, TRUE)
```
3) At predict time (script enforces):
- No missing/extra columns after alignment.
- Guardrail doesn’t trip (near-constant share ≤ 0.3).

4) Output sanity:
- Probs vary across games (not all equal or >0.99).
- Spot-check a few games vs. betting lines or naive baselines for realism.

---

## Developer ergonomics
- Keep `home_team`/`away_team` until **after** `matchup` is formed. The recipe will drop them in the baked matrix.
- Use `fs::dir_create()` before writes to avoid Windows path lock errors.
- Avoid hard-coded absolute paths; rely on `R/utils_io.R` (`read_paths()`, `read_params()`).

---

## Deliverables
Claude should produce a PR that:
1) Adds/updates `scripts/03_train_xgb.R` (train-time recipe + artifacts + model).
2) Rewrites `scripts/06_predict_upcoming.R` to bake with saved recipe and align columns.
3) Fixes `R/features_build.R` turnover/third-down logic if needed.
4) Ensures `scripts/02_build_features.R` emits `data_processed/train_raw.parquet` (raw features, no dummies).
5) Adds lightweight tests (see above) or at least asserts in scripts.
6) Includes a short README section **“Train & Predict Quickstart”** with the CLI sequence.

---

## Quickstart (CLI sequence to verify end-to-end)
```bash
# 1) Build features
Rscript scripts/02_build_features.R

# 2) Train (saves recipe, feature names, model)
Rscript scripts/03_train_xgb.R

# 3) Predict upcoming
Rscript scripts/06_predict_upcoming.R
```
Expected: `data/upcoming_predictions_wk<week>.parquet` + `.csv` with varied probabilities and a `matchup` column.

---

## TODO checklist for Claude
- [ ] Scan all `.R` files in `scripts/` and `R/` for any hand-dummying, imputation, or column `select()` that could desync train/predict; remove/centralize under **recipes**.
- [ ] Ensure `data_processed/train_raw.parquet` is created with `home_win`, `game_id`, `season`, `week`, and **raw** features (no manual dummies).
- [ ] Create `scripts/03_train_xgb.R` to: build recipe → `prep()` → `bake()` → save `rec_prepped.rds` + `xgb_feature_names.rds` → train XGB → save `winprob_xgb.model`.
- [ ] Refactor `scripts/06_predict_upcoming.R` to: build/read `upcoming_raw` → create `matchup` early → `bake()` with saved recipe → align to `x_cols` → predict → write outputs.
- [ ] Add guardrails: constant-column share check; confirm objective `binary:logistic`.
- [ ] Update `R/features_build.R` turnover and third-down logic; remove references to missing `turnover` fields.
- [ ] Add asserts (`stopifnot`) for required columns at key stages.
- [ ] Add README “Quickstart” with the 3-step run sequence.
- [ ] Push PR with clear commit messages and a summary of root-cause and fixes.
