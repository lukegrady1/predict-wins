# scripts/06_predict_upcoming.R
# Purpose: Predict upcoming REG games for params$season_current, params$week_cutoff
# using the trained xgboost model (models/winprob_xgb.rds), and render a pretty HTML slate.

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(arrow)
  library(glue)
  library(fs)
  library(nflreadr)
  library(tidymodels)
})

source("R/utils_io.R")
source("R/features_join.R")   # build_game_index(), join_side_features()

paths  <- read_paths()
params <- read_params()

set.seed(42)
fs::dir_create(paths$data_processed)
fs::dir_create("reports")

# ----------------------------- Load artifacts ---------------------------------
model_path <- fs::path("models", "winprob_xgb.rds")
if (!fs::file_exists(model_path)) {
  stop(glue("Missing model at {model_path}. Run scripts/04_train_model.R first."))
}
fit <- readRDS(model_path)

season <- params$season_current
week   <- params$week_cutoff

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

# Use same shape as training, then filter to week
game_idx <- build_game_index(sched_all) %>% dplyr::filter(week == !!week)

# ---------------------- Build pre-game team snapshots -------------------------
if (week <= 1) stop("week_cutoff <= 1: no prior games to build pre-week snapshots.")

snap <- feat_cur %>%
  dplyr::filter(season == !!season, week < !!week) %>%
  dplyr::arrange(team, dplyr::desc(week)) %>%
  dplyr::group_by(team) %>%
  dplyr::slice_head(n = 1) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(week = !!week)  # stamp to upcoming week for the join

# ------------------ Join snapshots to home/away and build frame ---------------
home_side <- join_side_features(game_idx = game_idx, features_tw = snap, side = "home")
away_side <- join_side_features(game_idx = game_idx, features_tw = snap, side = "away")

# Merge sides; be robust about which identifier columns exist
id_wishlist <- c("season","week","game_id","gameday","weekday","gametime",
                 "home_team","away_team","location","roof","surface")
ids_from_home <- intersect(id_wishlist, names(home_side))

upcoming <- home_side %>%
  dplyr::select(dplyr::all_of(ids_from_home), dplyr::starts_with("home_")) %>%
  dplyr::inner_join(
    away_side %>% dplyr::select(season, week, game_id, dplyr::starts_with("away_")),
    by = c("season","week","game_id")
  )

# ----------------------- Build the same *_diff features -----------------------
mk_diff <- function(df, a, b, name) {
  if (all(c(a, b) %in% names(df))) df[[name]] <- df[[a]] - df[[b]]
  df
}

upcoming <- upcoming %>%
  mk_diff("home_yards_per_play", "away_yards_per_play", "ypp_diff") %>%
  mk_diff("home_success_rate",   "away_success_rate",   "success_rate_diff") %>%
  mk_diff("home_pass_pct",       "away_pass_pct",       "pass_pct_diff") %>%
  mk_diff("home_opp_win_pct_pre","away_opp_win_pct_pre","sos_winpct_diff") %>%
  mk_diff("home_opp_pd_avg_pre", "away_opp_pd_avg_pre", "sos_pdavg_diff") %>%
  mk_diff("home_qb_change",      "away_qb_change",      "qb_change_diff")

# -------------------- Select numeric features used by the model ---------------
num_feature_candidates <- upcoming %>%
  dplyr::select(dplyr::starts_with("home_"), dplyr::starts_with("away_"), dplyr::ends_with("_diff")) %>%
  dplyr::select(where(is.numeric)) %>%
  names()
if (length(num_feature_candidates) == 0) stop("No numeric model features found in upcoming frame.")

newdata <- upcoming %>% dplyr::select(dplyr::all_of(num_feature_candidates))

# ----------------------------- Predict probabilities --------------------------
pred_prob <- predict(fit, new_data = newdata, type = "prob")
p <- if (".pred_yes" %in% names(pred_prob)) pred_prob$.pred_yes else pred_prob[[1]]

# Recompute the IDs that are actually present **now** on `upcoming`
present_ids <- intersect(id_wishlist, names(upcoming))

preds <- upcoming %>%
  dplyr::select(dplyr::all_of(present_ids)) %>%
  dplyr::mutate(p_home_win = p)

# Sort by kickoff if present; otherwise by home_team then game_id
if (all(c("gameday","gametime") %in% names(preds))) {
  preds <- preds %>% dplyr::arrange(season, week, gameday, gametime,
                                    dplyr::across(dplyr::all_of(intersect("home_team", names(.)))))
} else {
  preds <- preds %>% dplyr::arrange(season, week,
                                    dplyr::across(dplyr::all_of(intersect("home_team", names(.)))),
                                    game_id)
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
  dplyr::mutate(
    matchup = dplyr::case_when(
      !is.na(away_team) & !is.na(home_team) ~ paste0(away_team, " @ ", home_team),
      !is.na(home_team) ~ paste0("(away) @ ", home_team),
      !is.na(away_team) ~ paste0(away_team, " @ (home)"),
      TRUE ~ "(teams unavailable)"
    )
  ) %>%
  dplyr::arrange(dplyr::desc(p_home_win))

# HTML helpers
logo_img <- function(url, size = 28) {
  if (is.na(url) || !nzchar(url)) return("")
  sprintf('<img src="%s" alt="" width="%d" height="%d" style="vertical-align:middle;border-radius:4px;">', url, size, size)
}

rows_html <- purrr::pmap_chr(
  disp %>% dplyr::select(kickoff_date, kickoff_time, away_team, home_team, away_logo, home_logo, p_text),
  function(kickoff_date, kickoff_time, away_team, home_team, away_logo, home_logo, p_text) {
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
<p style="margin-top:16px;color:#666">Source: trained model (xgboost) and features_current.parquet.</p>
</body></html>'
)

out_html <- fs::path("reports", glue("upcoming_week_{week}.html"))
writeLines(html, con = out_html)
message(glue("Wrote pretty HTML slate: {fs::path_abs(out_html)}"))
