# scripts/06_predict_upcoming.R
# Purpose: Predict upcoming REG games using saved recipe and model artifacts
# Key fix: Use exact same preprocessing as training via saved recipe

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(arrow)
  library(glue)
  library(fs)
  library(nflreadr)
  library(recipes)
  library(xgboost)
})

source("R/utils_io.R")
source("R/features_join.R")

paths  <- read_paths()
params <- read_params()

set.seed(42)
fs::dir_create(paths$data_processed)
fs::dir_create("reports")

# ----------------------------- Load artifacts ---------------------------------
message("Loading training artifacts...")

# Load prepped recipe
rec_path <- fs::path("models", "rec_prepped.rds")
if (!fs::file_exists(rec_path)) {
  stop(glue("Missing recipe at {rec_path}. Run scripts/03_train_xgb.R first."))
}
rec <- readRDS(rec_path)

# Load feature names/order
feature_names_path <- fs::path("models", "xgb_feature_names.rds")
if (!fs::file_exists(feature_names_path)) {
  stop(glue("Missing feature names at {feature_names_path}. Run scripts/03_train_xgb.R first."))
}
x_cols <- readRDS(feature_names_path)

# Load trained model
model_path <- fs::path("models", "winprob_xgb.model")
if (!fs::file_exists(model_path)) {
  stop(glue("Missing model at {model_path}. Run scripts/03_train_xgb.R first."))
}
bst <- xgb.load(model_path)

# Verify model objective (if available)
attrs <- xgb.attributes(bst)
objective <- attrs$objective
if (!is.null(objective) && objective != "binary:logistic") {
  warning(glue("Model objective is '{objective}', expected 'binary:logistic'"))
} else if (is.null(objective)) {
  message("Model objective not stored in attributes - assuming binary:logistic")
}

message(glue("Loaded artifacts: recipe, {length(x_cols)} features, model"))

# ----------------------------- Build upcoming raw data --------------------------
season <- params$season_current
week   <- params$week_cutoff

message(glue("Building upcoming data for season {season}, week {week}"))

# Load current season features (contains team-week data through week_cutoff-1)
features_current_path <- fs::path(paths$data_processed, "features_current.parquet")
if (!fs::file_exists(features_current_path)) {
  stop(glue("Missing {features_current_path}. Run scripts/02_build_features.R first."))
}
feat_cur <- arrow::read_parquet(features_current_path) %>% clean_names()

# ------------------------- Build upcoming game index --------------------------
standardize_schedule_cols <- function(df) {
  df <- df %>% janitor::clean_names()
  if (!"game_type" %in% names(df)) {
    if ("season_type" %in% names(df)) df <- df %>% dplyr::rename(game_type = season_type)
    else if ("gametype" %in% names(df)) df <- df %>% dplyr::rename(game_type = gametype)
  }
  if (!"game_type" %in% names(df)) stop("`game_type` column is missing in schedules.")
  df %>% dplyr::mutate(game_type = toupper(as.character(game_type)),
                       week = suppressWarnings(as.integer(week)))
}

sched_all  <- nflreadr::load_schedules(seasons = season) %>% standardize_schedule_cols()
sched_week <- sched_all %>% dplyr::filter(game_type == "REG", week == !!week)
if (nrow(sched_week) == 0) stop(glue("No REG games found for season {season}, week {week}."))

# Build game index for upcoming week
game_idx <- build_game_index(sched_all) %>% dplyr::filter(week == !!week)

# ---------------------- Build pre-game team snapshots -------------------------
if (week <= 1) stop("week_cutoff <= 1: no prior games to build pre-week snapshots.")

# Get latest team snapshots (most recent game before the upcoming week)
snap <- feat_cur %>%
  dplyr::filter(season == !!season, week < !!week) %>%
  dplyr::arrange(team, dplyr::desc(week)) %>%
  dplyr::group_by(team) %>%
  dplyr::slice_head(n = 1) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(week = as.integer(!!week))  # stamp to upcoming week for the join (ensure integer type)

# ------------------ Join snapshots to home/away and build raw frame ---------------
home_side <- join_side_features(game_idx = game_idx, features_tw = snap, side = "home")
away_side <- join_side_features(game_idx = game_idx, features_tw = snap, side = "away")

# Merge sides while preserving team strings for matchup creation
id_wishlist <- c("season","week","game_id","gameday","weekday","gametime",
                 "home_team","away_team","location","roof","surface")
ids_from_home <- intersect(id_wishlist, names(home_side))

upcoming_raw <- home_side %>%
  dplyr::select(dplyr::all_of(ids_from_home), dplyr::starts_with("home_")) %>%
  dplyr::inner_join(
    away_side %>% dplyr::select(season, week, game_id, dplyr::starts_with("away_")),
    by = c("season","week","game_id")
  )

# **CRITICAL**: Create matchup BEFORE any processing that might drop team strings
# Fix column name conflicts from join (away_team.x is the correct one)
if ("away_team.x" %in% names(upcoming_raw) && !"away_team" %in% names(upcoming_raw)) {
  upcoming_raw <- upcoming_raw %>%
    dplyr::select(-any_of(c("away_team.y"))) %>%
    dplyr::rename(away_team = away_team.x)
}

upcoming_raw <- upcoming_raw %>%
  dplyr::mutate(
    matchup = paste0(away_team, " @ ", home_team)
  )

message(glue("Built upcoming_raw: {nrow(upcoming_raw)} games"))

# ----------------------------- Bake with saved recipe -------------------------
message("Applying saved recipe to upcoming data...")

# Bake upcoming data with the saved recipe
upcoming_baked <- bake(rec, new_data = upcoming_raw)

message(glue("Baked upcoming data: {nrow(upcoming_baked)} rows, {ncol(upcoming_baked)} columns"))

# ----------------------------- Align columns to training -------------------------
message("Aligning columns to training feature set...")

# Get current column names (excluding target if present)
current_cols <- setdiff(names(upcoming_baked), "home_win")

# Find missing columns and add them with zeros
missing_cols <- setdiff(x_cols, current_cols)
if (length(missing_cols) > 0) {
  message(glue("Adding {length(missing_cols)} missing columns with zeros: {paste(head(missing_cols, 3), collapse=', ')}..."))
  for (col in missing_cols) {
    upcoming_baked[[col]] <- 0
  }
}

# Find extra columns and drop them
extra_cols <- setdiff(current_cols, x_cols)
if (length(extra_cols) > 0) {
  message(glue("Dropping {length(extra_cols)} extra columns: {paste(head(extra_cols, 3), collapse=', ')}..."))
  upcoming_baked <- upcoming_baked %>% select(-all_of(extra_cols))
}

# Reorder columns to match training exactly
upcoming_baked <- upcoming_baked %>% select(all_of(x_cols))

# Verify exact match
stopifnot(identical(colnames(upcoming_baked), x_cols))
message("Column alignment verified!")

# ----------------------------- Guardrails ------------------------------------
message("Running guardrails...")

# Check for too many near-constant columns
nz <- vapply(upcoming_baked, function(x) length(unique(x)), integer(1))
constant_pct <- mean(nz <= 1)
# Be more permissive for early season (many features will be NA/0 due to limited data)
max_constant_pct <- if(week <= 4) 0.98 else 0.3  # Allow 98% constant features in early weeks
if (constant_pct > max_constant_pct) {
  # Show some examples of constant columns for debugging
  constant_cols <- names(upcoming_baked)[nz <= 1]
  message(glue("Constant columns ({length(constant_cols)} total): {paste(head(constant_cols, 5), collapse=', ')}..."))
  stop(glue("Too many near-constant columns in upcoming features ({round(100*constant_pct, 1)}% have ≤1 unique values); check joins and recipe."))
}
message(glue("Constant column check passed: {round(100*constant_pct, 1)}% columns with ≤1 unique values (threshold: {round(100*max_constant_pct, 1)}%)"))

# ----------------------------- Predict probabilities --------------------------
message("Generating predictions...")

# Convert to matrix and predict
upcoming_matrix <- data.matrix(upcoming_baked)
pred_probs <- predict(bst, upcoming_matrix)

# ----------------------------- Build output with identifiers ----------------
# Get identifiers that are actually present in upcoming_raw
present_ids <- intersect(c("season","week","game_id","gameday","weekday","gametime",
                          "home_team","away_team","location","roof","surface","matchup"),
                        names(upcoming_raw))

preds <- upcoming_raw %>%
  dplyr::select(dplyr::all_of(present_ids)) %>%
  dplyr::mutate(p_home_win = pred_probs)

# Sort by kickoff if present; otherwise by home_team then game_id
if (all(c("gameday","gametime") %in% names(preds))) {
  preds <- preds %>% dplyr::arrange(season, week, gameday, gametime,
                                    dplyr::across(dplyr::all_of(intersect("home_team", names(.)))))
} else {
  preds <- preds %>% dplyr::arrange(season, week,
                                    dplyr::across(dplyr::all_of(intersect("home_team", names(.)))),
                                    game_id)
}

# ----------------------------- Sanity checks on predictions ------------------
message("Validating predictions...")

pred_range <- range(preds$p_home_win)
pred_var <- var(preds$p_home_win)

message(glue("Prediction range: [{round(pred_range[1], 3)}, {round(pred_range[2], 3)}]"))
message(glue("Prediction variance: {round(pred_var, 6)}"))

# Check for suspicious constant probabilities
if (pred_var < 1e-6) {
  warning("Predictions appear nearly constant - check for feature alignment issues")
}

if (any(preds$p_home_win > 0.99)) {
  warning(glue("{sum(preds$p_home_win > 0.99)} predictions > 0.99 - check for issues"))
}

# ------------------------------- Save outputs ---------------------------------
out_parquet <- fs::path(paths$data_processed, glue("upcoming_predictions_wk{week}.parquet"))
out_csv     <- fs::path(paths$data_processed, glue("upcoming_predictions_wk{week}.csv"))

arrow::write_parquet(preds, out_parquet)
readr::write_csv(preds, out_csv)

message(glue("Wrote: {out_parquet} and {out_csv}"))

# --------------------------- Pretty HTML slate --------------------------------
# Guarantee columns exist for HTML (avoid case_when errors)
if (!"home_team" %in% names(preds)) preds$home_team <- NA_character_
if (!"away_team" %in% names(preds)) preds$away_team <- NA_character_
if (!"gameday"   %in% names(preds)) preds$gameday   <- NA_character_
if (!"gametime"  %in% names(preds)) preds$gametime  <- NA_character_
if (!"matchup"   %in% names(preds)) preds$matchup   <- "(matchup unavailable)"

# Logos
teams_df <- try(nflreadr::load_teams(), silent = TRUE)
if (inherits(teams_df, "try-error")) {
  teams_df <- tibble(team_abbr = character(), team_logo_espn = character())
}
teams_df <- teams_df %>% clean_names() %>%
  dplyr::select(team_abbr, team_logo_espn) %>%
  dplyr::rename(team = team_abbr, logo = team_logo_espn)

# Build display table
disp <- preds %>%
  dplyr::mutate(
    kickoff_date = dplyr::coalesce(gameday, as.character(week)),
    kickoff_time = dplyr::coalesce(gametime, ""),
    p_text  = sprintf("%.1f%%", 100 * p_home_win)
  ) %>%
  left_join(teams_df, by = c("home_team" = "team")) %>% dplyr::rename(home_logo = logo) %>%
  left_join(teams_df, by = c("away_team" = "team")) %>% dplyr::rename(away_logo = logo) %>%
  dplyr::arrange(dplyr::desc(p_home_win))

# HTML helpers
logo_img <- function(url, size = 28) {
  if (is.na(url) || !nzchar(url)) return("")
  sprintf('<img src="%s" alt="" width="%d" height="%d" style="vertical-align:middle;border-radius:4px;">', url, size, size)
}

rows_html <- purrr::pmap_chr(
  disp %>% dplyr::select(kickoff_date, kickoff_time, away_team, home_team, away_logo, home_logo, matchup, p_text),
  function(kickoff_date, kickoff_time, away_team, home_team, away_logo, home_logo, matchup, p_text) {
    sprintf(
      '<tr>
        <td style="padding:8px">%s</td>
        <td style="padding:8px">%s</td>
        <td style="padding:8px">%s %s&nbsp;&nbsp;<b>%s</b>&nbsp;&nbsp;%s %s</td>
        <td style="padding:8px;text-align:right;"><b>%s</b></td>
      </tr>',
      kickoff_date %||% "", kickoff_time %||% "",
      logo_img(away_logo), ifelse(is.na(away_team), "", away_team),
      "at",
      ifelse(is.na(home_team), "", home_team), logo_img(home_logo),
      p_text
    )
  }
)

html <- glue::glue(
  '<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Upcoming Week {week} — Win Probabilities</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;}}
h1{{margin:0 0 12px}} table{{border-collapse:collapse;width:100%;}}
th,td{{border-bottom:1px solid #eee;}} thead th{{text-align:left;padding:8px;color:#666;font-weight:600;}}
tr:hover{{background:#fafafa;}}
.badge{{background:#eef;border-radius:6px;padding:2px 8px;margin-left:6px;font-size:12px;color:#334;}}
</style></head>
<body>
<h1>Upcoming Games — Week {week} ({season}) <span class="badge">sorted by P(home)</span></h1>
<table>
  <thead><tr>
    <th>Kickoff Date</th><th>Kickoff Time</th><th>Matchup</th><th style="text-align:right">P(Home Win)</th>
  </tr></thead>
  <tbody>
    {paste(rows_html, collapse = "\n")}
  </tbody>
</table>
<p style="margin-top:16px;color:#666">Source: Recipe-based XGBoost model with {length(x_cols)} features.</p>
</body></html>'
)

out_html <- fs::path("reports", glue("upcoming_week_{week}.html"))
writeLines(html, con = out_html)
message(glue("Wrote pretty HTML slate: {fs::path_abs(out_html)}"))

# ----------------------------- Final summary ----------------------------------
message("=== Prediction Summary ===")
message(glue("Games predicted: {nrow(preds)}"))
message(glue("Prediction range: [{round(pred_range[1], 3)}, {round(pred_range[2], 3)}]"))
message(glue("Features used: {length(x_cols)}"))
message(glue("Outputs: {out_parquet}, {out_csv}, {out_html}"))
message("Prediction completed successfully!")