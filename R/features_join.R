# R/features_join.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

# --- Game index (REG schedule only) -------------------------------------------
build_game_index <- function(sched) {
  sched %>%
    clean_names() %>%
    filter(game_type == "REG") %>%
    select(season, week, game_id, gameday, weekday, gametime,
           home_team, away_team, location, roof, surface)
}

# --- Join helper: attach team-week features to home/away sides ----------------
join_side_features <- function(game_idx, features_tw, side = c("home", "away")) {
  side   <- match.arg(side)
  key_col <- if (side == "home") "home_team" else "away_team"
  prefix  <- paste0(side, "_")
  
  # Build a programmatic named 'by' vector: names = LHS (game_idx), values = RHS (features_tw)
  by_keys <- c(season = "season", week = "week", game_id = "game_id")
  by_keys[[key_col]] <- "team"  # e.g., by_keys["home_team"] = "team"
  
  df <- dplyr::left_join(game_idx, features_tw, by = by_keys)
  
  # Keep schedule keys unprefixed; prefix everything else from the features side
  keep_keys <- c(
    "season","week","game_id","gameday","weekday","gametime",
    "home_team","away_team","location","roof","surface"
  )
  value_cols <- setdiff(names(df), keep_keys)
  
  df %>%
    dplyr::rename_with(~ paste0(prefix, .x), dplyr::all_of(value_cols))
}

# --- Build a single model frame (game-level) from features + schedule ----------
# Returns one row per game with home_* and away_* columns and `home_win` target.
build_model_frame <- function(sched, features_tw) {
  game_idx <- build_game_index(sched)
  
  home_side <- join_side_features(game_idx, features_tw, side = "home")
  away_side <- join_side_features(game_idx, features_tw, side = "away")
  
  # Merge sides: keep schedule context + all home_* then add away_*
  mf <- home_side %>%
    select(
      season, week, game_id, gameday, weekday, gametime,
      home_team, away_team, location, roof, surface,
      starts_with("home_")
    ) %>%
    inner_join(
      away_side %>% select(season, week, game_id, starts_with("away_")),
      by = c("season","week","game_id")
    )
  
  # Target derived safely; no fragile opponent check
  mf <- mf %>%
    mutate(
      home_win = coalesce(home_win, as.integer(home_point_diff > 0))
    )
  
  # (Optional) If both columns exist, you can keep a quiet check without erroring:
  if (all(c("home_opponent", "away_team") %in% names(mf))) {
    mf <- mf %>%
      filter(is.na(home_opponent) | home_opponent == away_team)
  }
  
  mf
}


# --- Convenience: file writers (use your utils_io helpers if available) -------
# These assume the caller provides per-season schedule and features for training.
write_model_frame_one_season <- function(mf, out_path) {
  # If utils_io::write_parquet_safe exists, prefer it; else fallback
  if ("write_parquet_safe" %in% getNamespaceExports("utils")) {
    utils::write_parquet_safe(mf, out_path)
  } else {
    arrow::write_parquet(mf, out_path)
  }
}

# --- (Safe) column ordering for readability -----------------------------------
order_model_frame_cols <- function(mf) {
  # Desired groups
  keys   <- c("season","week","game_id","gameday","weekday","gametime",
              "home_team","away_team","location","roof","surface")
  target <- c("home_win")
  
  # Compute present columns
  present_keys   <- intersect(keys,   names(mf))
  present_target <- intersect(target, names(mf))
  homec  <- sort(grep("^home_", names(mf),  value = TRUE))
  awayc  <- sort(grep("^away_", names(mf),  value = TRUE))
  other  <- setdiff(names(mf), c(present_keys, present_target, homec, awayc))
  
  dplyr::select(mf, dplyr::all_of(present_keys),
                dplyr::all_of(present_target),
                dplyr::all_of(homec),
                dplyr::all_of(awayc),
                dplyr::all_of(other))
}

