# R/predict_upcoming.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(arrow)
  library(jsonlite)
  library(nflreadr)
  library(janitor)
})

# Build "home vs away" rows for week >= target_week with feature columns
# exactly matching the model's expected inputs.
build_upcoming_matchups <- function(season, target_week,
                                    features_current_path = "data/features_current.parquet",
                                    feature_info_path = "models/feature_info.json") {
  
  stopifnot(file.exists(features_current_path), file.exists(feature_info_path))
  
  feats <- arrow::read_parquet(features_current_path)
  info  <- jsonlite::fromJSON(feature_info_path)
  
  # Expected predictor names (right-hand side of formula)
  expected <- setdiff(info$features, c("win"))  # make sure we don't include outcome
  
  # Load schedule and filter upcoming
  sched <- nflreadr::load_schedules(seasons = season) %>%
    clean_names() %>%
    filter(game_type == "REG", week >= target_week) %>%
    mutate(game_day = as_date(gameday)) %>%
    select(season, week, game_id, game_day, home_team, away_team, everything())
  
  # Prepare home/away features from team snapshot (through week_cutoff - 1)
  home_feats <- feats %>%
    rename(home_team = team) %>%
    rename_with(~paste0("home_", .x), -c(season, home_team))
  
  away_feats <- feats %>%
    rename(away_team = team) %>%
    rename_with(~paste0("away_", .x), -c(season, away_team))
  
  upcoming <- sched %>%
    left_join(home_feats, by = c("season","home_team")) %>%
    left_join(away_feats, by = c("season","away_team")) %>%
    mutate(
      # Rest days based on last_game_day in features vs game_day in schedule
      home_rest_days = as.numeric(game_day - home_last_game_day),
      away_rest_days = as.numeric(game_day - away_last_game_day)
    )
  
  # Ensure all expected columns exist; if missing, add with safe defaults (0)
  missing_cols <- setdiff(expected, names(upcoming))
  if (length(missing_cols) > 0) {
    message("Filling missing predictors with 0: ", paste(missing_cols, collapse = ", "))
    for (mc in missing_cols) upcoming[[mc]] <- 0
  }
  
  # Keep only expected predictors + id info
  keep_cols <- c("season","week","game_id","game_day","home_team","away_team", expected)
  upcoming %>% select(any_of(keep_cols))
}

# Predict probability of home team win for each upcoming game
predict_upcoming_games <- function(fit, season, target_week,
                                   features_current_path = "data/features_current.parquet",
                                   feature_info_path = "models/feature_info.json") {
  
  newdat <- build_upcoming_matchups(
    season = season,
    target_week = target_week,
    features_current_path = features_current_path,
    feature_info_path = feature_info_path
  )
  
  # Predictions
  probs <- predict(fit, newdat, type = "prob") %>%
    bind_cols(newdat %>% select(season, week, game_day, home_team, away_team))
  
  # Standard tidy output
  probs %>%
    transmute(
      season, week, game_day, home_team, away_team,
      home_win_prob = .pred_win
    ) %>%
    arrange(week, game_day, home_team)
}
