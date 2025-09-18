# scripts/04_train_model.R
# Purpose: train logistic-regression model on data/model_frame.parquet

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(glue)
  library(fs)
})

source("R/utils_io.R")
source("R/modeling.R")

paths <- read_paths()

mf_path <- fs::path(paths$data_processed, "model_frame.parquet")
stopifnot(fs::file_exists(mf_path))

df <- read_parquet_safe(mf_path)

# Ensure outcome factor
outcome <- "win"
df <- coerce_outcome_factor(df, outcome)

# Fit & evaluate
res <- fit_and_eval(df, outcome = outcome, test_prop = 0.2, seed = 123)

# Collect feature names from the prepped recipe
# (After preparation, the recipe stores terms; but here we just parse the formula)
feature_cols <- all.vars(stats::terms(as.formula(res$formula)))[-1]

# Save artifacts
save_artifacts(
  fit        = res$fit,
  formula    = res$formula,
  metrics    = res$metrics,
  feature_cols = feature_cols,
  models_dir = read_paths()$models_dir
)

message("Training complete.")
message(glue("Formula used: {res$formula}"))

# Print metrics to console
acc <- res$metrics$accuracy$.estimate
auc <- res$metrics$roc_auc$.estimate
message(glue("Accuracy: {round(acc, 3)} | ROC AUC: {round(auc, 3)}"))

# Optional quick peek of columns used
message("Features used:")
print(feature_cols)
