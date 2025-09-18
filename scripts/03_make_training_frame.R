# scripts/03_make_training_frame.R
# Purpose: Join team-week features into game-level (home/away) model frames for training
#          and produce a current-season snapshot for inference.

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(glue)
  library(fs)
  library(nflreadr)
  library(janitor)
})

source("R/utils_io.R")
source("R/features_join.R")  # provides build_model_frame() and order_model_frame_cols()

paths  <- read_paths()
params <- read_params()

ensure_dirs(paths$data_processed)

# ---------- Helpers (same schedule normalizer used in 02_build_features.R) ----------
standardize_schedule_cols <- function(df) {
  df <- df %>% janitor::clean_names()
  if (!"game_type" %in% names(df)) {
    if ("season_type" %in% names(df)) {
      message("Renaming season_type -> game_type")
      df <- df %>% dplyr::rename(game_type = season_type)
    } else if ("gametype" %in% names(df)) {
      message("Renaming gametype -> game_type")
      df <- df %>% dplyr::rename(game_type = gametype)
    } else if ("GameType" %in% names(df)) {
      message("Renaming GameType -> game_type")
      df <- df %>% dplyr::rename(game_type = GameType)
    }
  }
  if (!"game_type" %in% names(df)) {
    message("Columns present in schedules: "); print(names(df))
    stop("`game_type` column is missing in schedules.")
  }
  df %>%
    dplyr::mutate(
      game_type = toupper(as.character(game_type)),
      week      = suppressWarnings(as.integer(week))
    )
}

read_sched_season <- function(season) {
  nflreadr::load_schedules(seasons = season) %>% standardize_schedule_cols()
}

# ---------- Inputs ----------
train_seasons  <- params$seasons_train
season_current <- params$season_current
week_cutoff    <- params$week_cutoff

message(glue("[03_make_training_frame] Seasons: {paste(train_seasons, collapse=', ')}"))

# ---------- TRAIN: per-season model frames ----------
model_frames <- vector("list", length(train_seasons))
names(model_frames) <- as.character(train_seasons)

for (i in seq_along(train_seasons)) {
  sea <- train_seasons[[i]]
  
  # Read schedule and per-team features produced by 02_build_features.R
  sched <- read_sched_season(sea)
  
  feats_path <- fs::path(paths$data_processed, glue("features_train_{sea}.parquet"))
  if (!fs::file_exists(feats_path)) {
    stop(glue("Missing {feats_path}. Run scripts/02_build_features.R first."))
  }
  feats <- arrow::read_parquet(feats_path)
  
  # Build and write model frame
  mf <- build_model_frame(sched, feats) %>%
    order_model_frame_cols()
  
  out_mf_path <- fs::path(paths$data_processed, glue("model_frame_train_{sea}.parquet"))
  write_parquet_safe(mf, out_mf_path)
  message(glue("[03_make_training_frame] Wrote per-season model frame: {out_mf_path}  (n={nrow(mf)})"))
  
  model_frames[[i]] <- mf
}

# ---------- TRAIN: combined model frame ----------
model_frame_all <- dplyr::bind_rows(model_frames)
out_all_mf <- fs::path(
  paths$data_processed,
  glue("model_frame_train_{min(train_seasons)}_{max(train_seasons)}.parquet")
)
write_parquet_safe(model_frame_all, out_all_mf)
message(glue("[03_make_training_frame] Wrote combined model frame: {out_all_mf}  (n={nrow(model_frame_all)})"))

# ---------- CURRENT snapshot (through week_cutoff - 1) ----------
message(glue("[03_make_training_frame] Current season={season_current}, through week<{week_cutoff}"))

sched_cur <- read_sched_season(season_current)

feats_cur_path <- fs::path(paths$data_processed, "features_current.parquet")
if (!fs::file_exists(feats_cur_path)) {
  stop(glue("Missing {feats_cur_path}. Run scripts/02_build_features.R first."))
}
features_current <- arrow::read_parquet(feats_cur_path)

mf_current <- build_model_frame(sched_cur, features_current) %>%
  dplyr::filter(week < week_cutoff) %>%
  order_model_frame_cols()

out_cur_mf <- fs::path(paths$data_processed, "model_frame_current.parquet")
write_parquet_safe(mf_current, out_cur_mf)
message(glue("[03_make_training_frame] Wrote current snapshot model frame: {out_cur_mf}  (n={nrow(mf_current)})"))

message("[03_make_training_frame] Done.")
