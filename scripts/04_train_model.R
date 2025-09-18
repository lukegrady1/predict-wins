# scripts/04_train_model.R
# Purpose: Train a final win-probability model on all training seasons.
# Output: models/winprob_xgb.rds (final model)

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(arrow)
  library(glue)
  library(fs)
  library(tidymodels)
})

source("R/utils_io.R")
paths  <- read_paths()
params <- read_params()

set.seed(42)
ensure_dirs(paths$data_processed)
fs::dir_create("models")

# ------------------ Load model frame ------------------
mf_path_combined <- fs::path(
  paths$data_processed,
  glue("model_frame_train_{min(params$seasons_train)}_{max(params$seasons_train)}.parquet")
)
if (!fs::file_exists(mf_path_combined)) {
  stop(glue("Missing model frame {mf_path_combined}. Run 03_make_training_frame.R first."))
}

mf <- arrow::read_parquet(mf_path_combined) %>% clean_names()

# ------------------ Build numeric diffs ------------------
mk_diff <- function(df, a, b, name) {
  if (all(c(a, b) %in% names(df))) df[[name]] <- df[[a]] - df[[b]]
  df
}

mf <- mf %>%
  mk_diff("home_yards_per_play", "away_yards_per_play", "ypp_diff") %>%
  mk_diff("home_success_rate",   "away_success_rate",   "success_rate_diff") %>%
  mk_diff("home_pass_pct",       "away_pass_pct",       "pass_pct_diff") %>%
  mk_diff("home_opp_win_pct_pre","away_opp_win_pct_pre","sos_winpct_diff") %>%
  mk_diff("home_opp_pd_avg_pre", "away_opp_pd_avg_pre", "sos_pdavg_diff") %>%
  mk_diff("home_qb_change",      "away_qb_change",      "qb_change_diff") %>%
  mutate(home_win = as.factor(if_else(home_win == 1, "yes", "no")))

# ------------------ Select ONLY numeric model features ------------------
# Keep numeric columns that look like true features:
#   - start with "home_" or "away_"
#   - or end with "_diff"
num_feature_candidates <- mf %>%
  select(starts_with("home_"), starts_with("away_"), ends_with("_diff")) %>%
  select(where(is.numeric)) %>%
  names()

if (length(num_feature_candidates) == 0) {
  stop("No numeric model features found. Check earlier steps (01–03).")
}

mf_model <- mf %>%
  select(all_of(num_feature_candidates), home_win) %>%
  drop_na(home_win)  # safety

# ------------------ Model spec ------------------
rec <- recipe(home_win ~ ., data = mf_model) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors())

xgb_spec <- boost_tree(
  trees = 1000,
  learn_rate = 0.05,
  tree_depth = 6,
  min_n = 5,
  sample_size = 0.8
) %>%
  set_engine("xgboost", nthread = max(1, parallel::detectCores() - 1)) %>%
  set_mode("classification")

wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rec)

# ------------------ Train final model ------------------
final_fit <- fit(wf, data = mf_model)

final_model_path <- fs::path("models", "winprob_xgb.rds")
saveRDS(final_fit, final_model_path)
message(glue("[04_train_model] Saved final model: {final_model_path}"))
