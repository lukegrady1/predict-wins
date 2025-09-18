# scripts/03_train_xgb.R
# Purpose: Train XGBoost model using recipes for consistent preprocessing
# Saves: recipe, feature names, and trained model for predict-time consistency

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(glue)
  library(fs)
  library(recipes)
  library(xgboost)
})

source("R/utils_io.R")

paths  <- read_paths()
params <- read_params()

set.seed(42)
fs::dir_create("models")

# ---------- Load train_raw.parquet ----------
train_raw_path <- fs::path(paths$data_processed, "train_raw.parquet")
if (!fs::file_exists(train_raw_path)) {
  stop(glue("Missing {train_raw_path}. Run scripts/02_build_features.R first."))
}

train_raw <- arrow::read_parquet(train_raw_path)
message(glue("Loaded train_raw: {nrow(train_raw)} rows, {ncol(train_raw)} columns"))

# Filter out rows with missing home_win (games not yet played)
train_raw <- train_raw %>% filter(!is.na(home_win))
message(glue("After filtering NA home_win: {nrow(train_raw)} rows"))

# ---------- Build recipes pipeline ----------
message("Building recipes pipeline...")

# Create recipes pipeline before manual dummies
rec <- recipe(home_win ~ ., data = train_raw) %>%
  # Set ID roles for non-predictive columns
  update_role(game_id, season, week, new_role = "id") %>%
  # Remove ID columns from predictors
  step_rm(game_id, season, week) %>%
  # Impute missing numeric values
  step_impute_median(all_numeric_predictors()) %>%
  # Handle unknown levels in categorical variables
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  # Create dummy variables for all nominal predictors
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Remove zero-variance predictors
  step_zv(all_predictors())

# Prep the recipe
rec_prepped <- prep(rec, training = train_raw)
message("Recipe prepped successfully")

# Bake training data
train_baked <- bake(rec_prepped, new_data = train_raw)
message(glue("Baked training data: {nrow(train_baked)} rows, {ncol(train_baked)} columns"))

# Extract features (drop home_win target)
trainX <- train_baked %>% select(-home_win)
trainY <- train_baked$home_win

# Get exact feature names and order
x_cols <- colnames(trainX)
message(glue("Training features: {length(x_cols)} columns"))

# ---------- Train XGBoost model ----------
message("Training XGBoost model...")

# Convert to matrix for XGBoost
trainX_matrix <- data.matrix(trainX)

# Train XGBoost with very conservative parameters to produce realistic NFL probabilities
# With limited training data (only ~3300 games), we need aggressive regularization
bst <- xgboost(
  data = trainX_matrix,
  label = as.numeric(trainY),
  nrounds = 50,            # Very few rounds to prevent overfitting
  max_depth = 3,           # Shallow trees for generalization
  eta = 0.02,              # Very low learning rate for gradual learning
  subsample = 0.6,         # Strong subsampling for regularization
  colsample_bytree = 0.6,  # Feature subsampling for regularization
  reg_alpha = 1.0,         # Strong L1 regularization
  reg_lambda = 5.0,        # Very strong L2 regularization
  min_child_weight = 20,   # Higher minimum samples per leaf
  gamma = 0.1,             # Minimum loss reduction for splits
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbose = 1
)

message("XGBoost training completed")

# ---------- Save artifacts ----------
message("Saving training artifacts...")

# 1. Save prepped recipe
rec_path <- fs::path("models", "rec_prepped.rds")
saveRDS(rec_prepped, rec_path)
message(glue("Saved recipe: {rec_path}"))

# 2. Save exact feature names/order
feature_names_path <- fs::path("models", "xgb_feature_names.rds")
saveRDS(x_cols, feature_names_path)
message(glue("Saved feature names: {feature_names_path} ({length(x_cols)} features)"))

# 3. Save trained XGBoost model
model_path <- fs::path("models", "winprob_xgb.model")
xgb.save(bst, model_path)
message(glue("Saved XGBoost model: {model_path}"))

# ---------- Validation checks ----------
message("Running validation checks...")

# Check 1: Verify feature names were saved
stopifnot(length(readRDS(feature_names_path)) > 0)

# Check 2: Verify model objective
model_loaded <- xgb.load(model_path)
objective <- xgb.attributes(model_loaded)$objective
stopifnot(objective == "binary:logistic")

# Check 3: Test recipe can bake new data
test_bake <- bake(rec_prepped, new_data = train_raw[1:5, ])
stopifnot(identical(colnames(test_bake %>% select(-home_win)), x_cols))

message("All validation checks passed!")

# ---------- Summary ----------
message("=== Training Summary ===")
message(glue("Training samples: {nrow(train_raw)}"))
message(glue("Features: {length(x_cols)}"))
message(glue("Objective: {objective}"))
message(glue("Saved artifacts:"))
message(glue("  - Recipe: {rec_path}"))
message(glue("  - Feature names: {feature_names_path}"))
message(glue("  - Model: {model_path}"))

message("Training completed successfully!")