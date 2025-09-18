# NFL Win Probability Predictor

This repository builds NFL win probabilities using team performance features and XGBoost. The pipeline creates consistent features at train and predict time using recipes for robust preprocessing.

## Train & Predict Quickstart

Run these commands in sequence to build features, train the model, and generate predictions:

```bash
# 1) Build features
Rscript scripts/02_build_features.R

# 2) Train (saves recipe, feature names, model)
Rscript scripts/03_train_xgb.R

# 3) Predict upcoming
Rscript scripts/06_predict_upcoming.R
```

**Expected output**: `data/upcoming_predictions_wk<week>.parquet` + `.csv` with varied probabilities and a `matchup` column.

## Key Files

- **`scripts/02_build_features.R`**: Builds team-week features from schedules and play-by-play data. Creates `data/train_raw.parquet` with raw game-level features.
- **`scripts/03_train_xgb.R`**: Recipe-based preprocessing and XGBoost training. Saves consistent preprocessing pipeline and model artifacts.
- **`scripts/06_predict_upcoming.R`**: Uses saved recipe to preprocess upcoming games and generate predictions with proper column alignment.
- **`config/params.yml`**: Configure training seasons, current season, and week cutoff.
- **`config/paths.yml`**: Data directory paths.

## Configuration

Edit `config/params.yml` to adjust:
- `seasons_train`: Historical seasons for training
- `season_current`: Current season for predictions
- `week_cutoff`: Predict games >= this week using data < this week

## Architecture

The pipeline ensures train/predict consistency by:
1. **Raw features**: `02_build_features.R` creates game-level raw features without manual dummy coding
2. **Recipe-based preprocessing**: `03_train_xgb.R` uses `recipes` for imputation, dummy coding, and zero-variance removal
3. **Saved artifacts**: Recipe, feature names, and model are persisted for exact replication at predict-time
4. **Column alignment**: `06_predict_upcoming.R` aligns feature matrices exactly to training
5. **Guardrails**: Checks prevent silent regressions from constant features or misaligned columns