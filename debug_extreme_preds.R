source("R/utils_io.R")
source("R/features_join.R")
library(recipes)
library(dplyr)
library(arrow)

paths <- read_paths()
params <- read_params()

# Recreate the exact features that went into the model
feat_cur <- arrow::read_parquet("data/features_current.parquet")
sched_all <- nflreadr::load_schedules(seasons = 2025)
game_idx <- build_game_index(sched_all) %>% filter(week == 3)

# Build snapshots
snap <- feat_cur %>%
  filter(season == 2025, week < 3) %>%
  arrange(team, desc(week)) %>%
  group_by(team) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  mutate(week = as.integer(3))

# Join to create upcoming_raw exactly as in the prediction script
home_side <- join_side_features(game_idx = game_idx, features_tw = snap, side = "home")
away_side <- join_side_features(game_idx = game_idx, features_tw = snap, side = "away")

upcoming_raw <- home_side %>%
  select(season, week, game_id, gameday, weekday, gametime,
         home_team, away_team, location, roof, surface,
         starts_with("home_")) %>%
  inner_join(
    away_side %>% select(season, week, game_id, starts_with("away_")),
    by = c("season","week","game_id"),
    suffix = c("", "_from_away")
  ) %>%
  mutate(matchup = paste0(away_team, " @ ", home_team))

# Load the saved recipe and bake the data
rec <- readRDS("models/rec_prepped.rds")
upcoming_baked <- bake(rec, new_data = upcoming_raw)

# Load feature names and align
x_cols <- readRDS("models/xgb_feature_names.rds")
upcoming_baked <- upcoming_baked %>% select(all_of(x_cols))

print("Sample of key numeric features from upcoming_baked:")
key_features <- c("home_yards_per_play", "away_yards_per_play",
                  "home_success_rate", "away_success_rate",
                  "home_pass_pct", "away_pass_pct")

# Find which of these features exist in the baked data
existing_features <- intersect(key_features, names(upcoming_baked))
print(paste("Features found:", paste(existing_features, collapse=", ")))

if (length(existing_features) > 0) {
  print(upcoming_baked %>%
    select(all_of(existing_features)) %>%
    slice_head(n = 5))

  print("Summary statistics:")
  print(summary(upcoming_baked %>% select(all_of(existing_features))))
} else {
  print("No key features found. Checking for any numeric variation:")
  numeric_cols <- names(upcoming_baked)[sapply(upcoming_baked, is.numeric)]
  if (length(numeric_cols) > 0) {
    # Check variance of numeric columns
    variances <- sapply(upcoming_baked[numeric_cols], var, na.rm = TRUE)
    non_zero_var <- variances[variances > 1e-10 & !is.na(variances)]
    print(paste("Columns with non-zero variance:", length(non_zero_var), "out of", length(numeric_cols)))
    if (length(non_zero_var) > 0) {
      print("Top 10 columns with highest variance:")
      print(head(sort(non_zero_var, decreasing = TRUE), 10))
    }
  }
}

# Check if teams are getting their proper features
print("\\nTeam-specific features check:")
# Look for patterns in the raw data before baking
print("Home team yards per play from raw data:")
if ("home_yards_per_play" %in% names(upcoming_raw)) {
  team_ypp <- upcoming_raw %>%
    select(home_team, away_team, home_yards_per_play, away_yards_per_play) %>%
    slice_head(n = 8)
  print(team_ypp)
} else {
  print("home_yards_per_play not found in upcoming_raw")
}