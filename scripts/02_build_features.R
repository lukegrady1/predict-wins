# scripts/02_build_features.R
# Purpose: build rolling team features for training seasons and current season snapshot.

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(glue)
  library(fs)
  library(nflreadr)
  library(janitor)  # you call janitor::clean_names()
})

source("R/utils_io.R")
source("R/features_build.R")

paths  <- read_paths()
params <- read_params()

ensure_dirs(paths$data_processed)

# ---------- Helpers ----------
# Force presence of a lowercase snake_case `game_type` column (REG/POST/PRE)
standardize_schedule_cols <- function(df) {
  df <- df %>% janitor::clean_names()
  
  # If game_type is missing, try common variants
  if (!"game_type" %in% names(df)) {
    if ("season_type" %in% names(df)) {
      message("Renaming season_type -> game_type")
      df <- df %>% dplyr::rename(game_type = season_type)
    } else if ("gametype" %in% names(df)) {
      message("Renaming gametype -> game_type")
      df <- df %>% dplyr::rename(game_type = gametype)
    } else if ("GameType" %in% names(df)) { # rare mixed-case from other sources
      message("Renaming GameType -> game_type")
      df <- df %>% dplyr::rename(game_type = GameType)
    }
  }
  
  # Final sanity check with helpful diagnostics
  if (!"game_type" %in% names(df)) {
    message("Columns present in schedules: ")
    print(names(df))
    stop("`game_type` column is missing in schedules. Upstream step likely dropped or renamed it.")
  }
  
  # Normalize values just in case (nflreadr already uses REG/POST/PRE)
  df <- df %>%
    dplyr::mutate(
      game_type = toupper(as.character(game_type)),
      week = suppressWarnings(as.integer(week))
    )
  
  df
}

# ---------- Load cached raw data helpers ----------
read_sched_season <- function(season) {
  # Reload fresh from nflreadr and normalize column names
  nflreadr::load_schedules(seasons = season) %>%
    standardize_schedule_cols()
}

read_pbp_season <- function(season) {
  p <- fs::path(paths$data_pbp, glue("pbp_{season}.parquet"))
  if (!fs::file_exists(p)) {
    stop(glue("Missing {p}. Run scripts/01_fetch_data.R first."))
  }
  arrow::read_parquet(p)
}

# ---------- Build training features (one parquet per season + combined) ----------
train_seasons <- params$seasons_train
message(glue("Building training features for seasons: {paste(train_seasons, collapse=', ')}"))

train_list <- vector("list", length(train_seasons))
names(train_list) <- as.character(train_seasons)

for (i in seq_along(train_seasons)) {
  sea <- train_seasons[[i]]
  
  sched <- read_sched_season(sea)
  # quick QA: ensure REG rows exist
  # message(glue("Season {sea} REG games: {nrow(dplyr::filter(sched, .data$game_type=='REG'))}"))
  
  pbp   <- read_pbp_season(sea)
  
  feats <- build_training_features_one_season(sched, pbp)
  
  out_season_path <- fs::path(paths$data_processed, glue("features_train_{sea}.parquet"))
  write_parquet_safe(feats, out_season_path)
  message(glue("Wrote: {out_season_path}  (n={nrow(feats)})"))
  
  train_list[[i]] <- feats
}

# Combined (handy for later joins/QA)
features_train_all <- dplyr::bind_rows(train_list)
out_all_path <- fs::path(
  paths$data_processed,
  glue("features_train_{min(train_seasons)}_{max(train_seasons)}.parquet")
)
write_parquet_safe(features_train_all, out_all_path)
message(glue("Wrote: {out_all_path}  (n={nrow(features_train_all)})"))

# ---------- Build current-season snapshot through (week_cutoff - 1) ----------
season_current <- params$season_current
week_cutoff    <- params$week_cutoff

message(glue("Building current features: season={season_current}, through week<{week_cutoff}"))

sched_cur <- read_sched_season(season_current)
pbp_cur   <- if (is.numeric(week_cutoff) && week_cutoff > 1) read_pbp_season(season_current) else {
  tibble() # no pbp yet if cutoff <= 1
}

features_current <- build_current_features_through_week(
  sched = sched_cur,
  pbp   = pbp_cur,
  week_cutoff = week_cutoff
)

out_cur_path <- fs::path(paths$data_processed, "features_current.parquet")
write_parquet_safe(features_current, out_cur_path)
message(glue("Wrote: {out_cur_path}  (n={nrow(features_current)})"))

message("Done: 02_build_features.R")
