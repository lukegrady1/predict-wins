# scripts/03_make_training_frame.R
# Purpose: build the modeling dataset (game-level, with win outcome).

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(glue)
  library(fs)
  library(nflreadr)
})

source("R/utils_io.R")
source("R/features_build.R")
source("R/features_join.R")

paths  <- read_paths()
params <- read_params()

ensure_dirs(paths$data_processed)

train_seasons <- params$seasons_train

all_games <- vector("list", length(train_seasons))

for (i in seq_along(train_seasons)) {
  sea <- train_seasons[[i]]
  
  message(glue("Building modeling frame for season {sea}..."))
  
  sched <- nflreadr::load_schedules(seasons = sea) %>% janitor::clean_names()
  feats <- read_parquet_safe(fs::path(paths$data_processed, glue("features_train_{sea}.parquet")))
  
  game_frame <- make_game_level_frame(sched, feats) %>%
    mutate(season = sea)
  
  all_games[[i]] <- game_frame
}

model_frame <- bind_rows(all_games)

out_path <- fs::path(paths$data_processed, "model_frame.parquet")
write_parquet_safe(model_frame, out_path)

message(glue("Wrote: {out_path}  (n={nrow(model_frame)})"))
