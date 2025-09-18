# scripts/06_predict_upcoming.R
# Purpose: predict upcoming games from saved model

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(fs)
  library(jsonlite)
})

source("R/utils_io.R")
source("R/predict_upcoming.R")

paths  <- read_paths()
params <- read_params()

fit <- readRDS(file.path(paths$models_dir, "fit_lr.rds"))

season <- params$season_current
week   <- params$week_cutoff

preds <- predict_upcoming_games(
  fit = fit,
  season = season,
  target_week = week,
  features_current_path = file.path(paths$data_processed, "features_current.parquet"),
  feature_info_path     = file.path(paths$models_dir, "feature_info.json")
)

# Save outputs
out_parquet <- file.path(paths$data_processed, glue::glue("upcoming_predictions_wk{week}.parquet"))
out_csv     <- file.path(paths$data_processed, glue::glue("upcoming_predictions_wk{week}.csv"))

arrow::write_parquet(preds, out_parquet)
readr::write_csv(preds, out_csv)

message(glue::glue("Wrote: {out_parquet} and {out_csv}"))
