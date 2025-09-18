# scripts/05_eval_model.R
# Purpose: deeper evaluation (confusion matrix, calibration) saved to reports/

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(fs)
  library(jsonlite)
  library(ggplot2)
})

source("R/utils_io.R")
source("R/modeling.R")
source("R/evaluate.R")

paths <- read_paths()

# Load data and model
mf_path <- fs::path(paths$data_processed, "model_frame.parquet")
stopifnot(fs::file_exists(mf_path))
df <- read_parquet_safe(mf_path)
df <- coerce_outcome_factor(df, "win")

fit <- readRDS(file.path(paths$models_dir, "fit_lr.rds"))

# Evaluate
res <- evaluate_holdout(fit, df, outcome = "win", test_prop = 0.2, seed = 123)

# Save metrics as JSON
metrics_tbl <- bind_rows(
  res$metrics$accuracy %>% as_tibble() %>% mutate(type = "accuracy"),
  res$metrics$roc_auc  %>% as_tibble() %>% mutate(type = "roc_auc")
)
jsonlite::write_json(metrics_tbl, file.path(paths$models_dir, "metrics_detailed.json"),
                     pretty = TRUE, auto_unbox = TRUE)

# Ensure reports dir
if (!dir.exists("reports")) dir.create("reports", recursive = TRUE)

# Confusion matrix plot
cm <- res$conf_mat %>% autoplot(type = "heatmap")
ggsave(filename = "reports/confusion_matrix.png", plot = cm, width = 6, height = 5, dpi = 150)

# Calibration plot
cal_plot <- ggplot(res$calibration, aes(x = mean_pred, y = emp_win)) +
  geom_point() + geom_abline(linetype = 2) +
  labs(title = "Calibration (by decile)", x = "Mean predicted P(win)", y = "Empirical win rate")
ggsave(filename = "reports/calibration.png", plot = cal_plot, width = 6, height = 5, dpi = 150)

message("Saved detailed metrics and plots to reports/ and models/")
