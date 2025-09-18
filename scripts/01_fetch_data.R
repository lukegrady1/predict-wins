# scripts/01_fetch_data.R
# Purpose: download/cache schedules and play-by-play to data-raw/

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(nflreadr)  # schedules
  library(nflfastR)  # play-by-play
  library(glue)
  library(fs)
})

source("R/utils_io.R")

paths  <- read_paths()
params <- read_params()

ensure_dirs(paths$data_raw, paths$data_pbp, paths$data_schedules)

# ------------------ Fetch Schedules ------------------
fetch_schedules <- function(seasons) {
  message(glue("Fetching schedules for seasons: {paste(seasons, collapse=', ')}"))
  sched <- nflreadr::load_schedules(seasons = seasons) %>%
    janitor::clean_names() %>%
    mutate(game_day = as_date(gameday))
  
  out_path <- fs::path(paths$data_schedules, glue("schedules_{min(seasons)}_{max(seasons)}.parquet"))
  write_parquet_safe(sched, out_path)
  message(glue("Wrote: {out_path}  (n={nrow(sched)})"))
  invisible(sched)
}

# ------------------ Fetch PBP ------------------------
# Writes one parquet per season to keep files manageable
fetch_pbp_by_season <- function(seasons) {
  purrr::walk(seasons, function(sea) {
    out_path <- fs::path(paths$data_pbp, glue("pbp_{sea}.parquet"))
    if (fs::file_exists(out_path)) {
      message(glue("Already cached: {out_path}"))
      return(invisible(NULL))
    }
    message(glue("Loading pbp for {sea}... (first time can take a bit)"))
    pbp <- nflfastR::load_pbp(sea)
    write_parquet_safe(pbp, out_path)
    message(glue("Wrote: {out_path}  (n={nrow(pbp)})"))
  })
}

# ------------------ Run ------------------------------
# 1) Training schedules (historical)
fetch_schedules(params$seasons_train)

# 2) Current season schedule (so we can predict upcoming weeks later)
fetch_schedules(params$season_current)

# 3) Play-by-play for training seasons
fetch_pbp_by_season(params$seasons_train)

# 4) (Optional) PBP for current season if week_cutoff > 1
if (is.numeric(params$week_cutoff) && params$week_cutoff > 1) {
  fetch_pbp_by_season(params$season_current)
}

message("Done: 01_fetch_data.R")
