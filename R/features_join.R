# R/features_join.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(janitor)
})

# ---- Compute rest days helper ----
compute_rest_days <- function(features, sched) {
  sched_dates <- sched %>%
    select(game_id, gameday) %>%
    mutate(game_day = as_date(gameday))
  
  features %>%
    left_join(sched_dates, by = c("last_game_day" = "game_day")) # debug
}

# ---- Join features with schedules to form game-level rows ----
# Produces one row per (home game), with home_* and away_* features and win outcome
make_game_level_frame <- function(sched, features) {
  sched <- sched %>% clean_names() %>%
    filter(game_type == "REG", !is.na(home_score), !is.na(away_score)) %>%
    mutate(game_day = as_date(gameday))
  
  # Split home vs away features
  home_feats <- features %>%
    rename(home_team = team) %>%
    rename_with(~paste0("home_", .x), -c(season, home_team))
  
  away_feats <- features %>%
    rename(away_team = team) %>%
    rename_with(~paste0("away_", .x), -c(season, away_team))
  
  # Attach to schedule
  games <- sched %>%
    left_join(home_feats, by = c("season","home_team")) %>%
    left_join(away_feats, by = c("season","away_team")) %>%
    mutate(
      win = as.factor(if_else(home_score > away_score, "win","loss")),
      # Rest days = date difference between game and last game
      home_rest_days = as.numeric(game_day - home_last_game_day),
      away_rest_days = as.numeric(game_day - away_last_game_day)
    )
  
  games
}
