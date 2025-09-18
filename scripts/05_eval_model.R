# scripts/05_eval_model.R
# Purpose: Evaluate model with rolling-by-season CV.
# Outputs:
#   - data_processed/metrics_training.csv
#   - data_processed/predictions_training.parquet

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
num_feature_candidates <- mf %>%
  select(starts_with("home_"), starts_with("away_"), ends_with("_diff")) %>%
  select(where(is.numeric)) %>%
  names()
if (length(num_feature_candidates) == 0) {
  stop("No numeric model features found. Check earlier steps (01–03).")
}

# Keep whatever identifiers actually exist
id_candidates <- c("season","week","game_id","home_team","away_team")
present_ids   <- intersect(id_candidates, names(mf))

mf_model <- mf %>%
  select(all_of(present_ids), all_of(num_feature_candidates), home_win) %>%
  drop_na(home_win)

# ------------------ Model spec ------------------
rec <- recipe(home_win ~ ., data = mf_model %>% select(all_of(num_feature_candidates), home_win)) %>%
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

# ------------------ Rolling-by-season evaluation ------------------
train_seasons <- sort(unique(mf_model$season))
if (length(train_seasons) < 2) stop("Need ≥2 seasons for evaluation")

all_preds  <- list()
all_metrics <- list()

for (s in train_seasons[-1]) {
  train_dat <- mf_model %>% filter(season < s)
  test_dat  <- mf_model %>% filter(season == s)
  
  fit <- fit(wf, data = train_dat)
  
  # Only bind identifiers that exist
  ids_in_test <- intersect(present_ids, names(test_dat))
  
  preds <- predict(fit, test_dat, type = "prob") %>%
    bind_cols(predict(fit, test_dat, type = "class")) %>%
    bind_cols(test_dat %>% select(all_of(ids_in_test), home_win))
  
  names(preds)[names(preds) == ".pred_yes"]   <- "p_home_win"
  names(preds)[names(preds) == ".pred_class"] <- "pred_class"
  
  auc_val <- roc_auc_vec(truth = preds$home_win, estimate = preds$p_home_win, event_level = "second")
  
  all_preds[[as.character(s)]]   <- preds
  all_metrics[[as.character(s)]] <- tibble(season = s, roc_auc = auc_val)
  message(glue("[05_eval_model] Season {s}: ROC AUC = {round(auc_val, 4)}"))
}

preds_df   <- bind_rows(all_preds)
metrics_df <- bind_rows(all_metrics) %>% arrange(season)

# Overall AUC
overall_auc <- roc_auc_vec(truth = preds_df$home_win, estimate = preds_df$p_home_win, event_level = "second")
metrics_df <- bind_rows(metrics_df, tibble(season = NA, roc_auc = overall_auc))


# ------------------ Save artifacts ------------------
metrics_path <- fs::path(paths$data_processed, "metrics_training.csv")
readr::write_csv(metrics_df, metrics_path)
message(glue("[05_eval_model] Wrote metrics: {metrics_path}"))

preds_path <- fs::path(paths$data_processed, "predictions_training.parquet")
write_parquet_safe(preds_df, preds_path)
message(glue("[05_eval_model] Wrote predictions: {preds_path}"))

message("[05_eval_model] Done.")
