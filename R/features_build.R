# R/features_build.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(janitor)
})

# ---- Team-game summaries from play-by-play ----
summarize_team_game <- function(pbp) {
  pbp %>%
    filter(!is.na(game_id), !is.na(week), !is.na(season)) %>%
    mutate(
      is_third_down_att  = (down == 3L) & play_type %in% c("pass","run"),
      is_third_down_conv = coalesce(third_down_converted, FALSE) & is_third_down_att,
      # turnover column doesn't exist in nflfastR pbp; use INT or fumble lost
      is_turnover        = coalesce(interception, FALSE) | coalesce(fumble_lost, FALSE),
      # scoring play: TD or made FG (sp sometimes marks scoring; keep it guarded)
      is_scoring_play    = coalesce((touchdown == 1) | (field_goal_result == "made") | (sp == 1), FALSE)
    )%>%
    group_by(game_id, season, week, team = posteam) %>%
    summarise(
      plays                    = n(),
      epa_per_play             = mean(epa, na.rm = TRUE),
      qb_epa                   = mean(qb_epa, na.rm = TRUE),
      third_down_att           = sum(is_third_down_att, na.rm = TRUE),
      third_down_conv          = sum(is_third_down_conv, na.rm = TRUE),
      third_down_pct           = dplyr::if_else(third_down_att > 0,
                                                third_down_conv / third_down_att, 0),
      turnovers                = sum(is_turnover, na.rm = TRUE),
      scoring_plays            = sum(is_scoring_play, na.rm = TRUE),
      .groups = "drop"
    )
}

# ---- Team records & dates from schedules (completed games only) ----
summarize_team_records <- function(sched_completed) {
  home <- sched_completed %>%
    transmute(
      season, week, game_id, game_day = as_date(gameday),
      team   = home_team,
      opp    = away_team,
      is_home = TRUE,
      win     = as.integer(home_score > away_score)
    )
  
  away <- sched_completed %>%
    transmute(
      season, week, game_id, game_day = as_date(gameday),
      team   = away_team,
      opp    = home_team,
      is_home = FALSE,
      win     = as.integer(away_score > home_score)
    )
  
  bind_rows(home, away) %>%
    arrange(season, week)
}

# ---- Rolling per-team features through a cutoff week (exclusive) ----
build_team_features_through <- function(sched, pbp, through_week = Inf) {
  sched <- sched %>% clean_names()
  
  # Schedules use game_type
  completed <- sched %>%
    filter(game_type == "REG",
           !is.na(home_score), !is.na(away_score),
           week < through_week)
  
  rec_long <- summarize_team_records(completed)
  
  rec_roll <- rec_long %>%
    group_by(season, team) %>%
    summarise(
      games_played   = n(),
      win_pct        = ifelse(games_played > 0, mean(win), 0),
      home_win_pct   = ifelse(sum(is_home) > 0, mean(win[is_home]), 0),
      away_win_pct   = ifelse(sum(!is_home) > 0, mean(win[!is_home]), 0),
      last_game_day  = ifelse(games_played > 0, max(game_day), as.Date(NA)),
      .groups = "drop"
    )
  
  # PBP uses season_type
  if (is.finite(through_week)) {
    pbp_cut <- pbp %>% filter(season_type == "REG", week < through_week)
  } else {
    pbp_cut <- pbp %>% filter(season_type == "REG")
  }
  
  team_off <- if (nrow(pbp_cut) > 0) {
    summarize_team_game(pbp_cut) %>%
      group_by(season, team) %>%
      summarise(
        epa_per_play            = mean(epa_per_play, na.rm = TRUE),
        qb_epa                  = mean(qb_epa, na.rm = TRUE),
        third_down_pct          = mean(third_down_pct, na.rm = TRUE),
        turnovers_per_game      = mean(turnovers, na.rm = TRUE),
        scoring_plays_per_game  = mean(scoring_plays, na.rm = TRUE),
        .groups = "drop"
      )
  } else {
    tibble(season = integer(), team = character(),
           epa_per_play = numeric(), qb_epa = numeric(),
           third_down_pct = numeric(), turnovers_per_game = numeric(),
           scoring_plays_per_game = numeric())
  }
  
  league_avgs <- team_off %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>%
    mutate(across(everything(), ~ ifelse(is.finite(.), ., 0)))
  
  all_teams <- tibble(season = unique(sched$season),
                      team   = sort(unique(c(sched$home_team, sched$away_team))))
  
  features <- all_teams %>%
    left_join(rec_roll, by = c("season","team")) %>%
    left_join(team_off, by = c("season","team")) %>%
    mutate(
      games_played  = replace_na(games_played, 0),
      win_pct       = replace_na(win_pct, 0),
      home_win_pct  = replace_na(home_win_pct, 0),
      away_win_pct  = replace_na(away_win_pct, 0),
      last_game_day = suppressWarnings(as_date(last_game_day)),
      epa_per_play           = coalesce(epa_per_play, league_avgs$epa_per_play),
      qb_epa                 = coalesce(qb_epa, league_avgs$qb_epa),
      third_down_pct         = coalesce(third_down_pct, league_avgs$third_down_pct),
      turnovers_per_game     = coalesce(turnovers_per_game, league_avgs$turnovers_per_game),
      scoring_plays_per_game = coalesce(scoring_plays_per_game, league_avgs$scoring_plays_per_game)
    )
  
  features
}

build_training_features_one_season <- function(sched, pbp) {
  build_team_features_through(sched, pbp, through_week = Inf)
}

build_current_features_through_week <- function(sched, pbp, week_cutoff) {
  wk <- ifelse(is.finite(week_cutoff), as.numeric(week_cutoff), Inf)
  if (wk <= 1) {
    all_teams <- tibble(season = unique(sched$season),
                        team   = sort(unique(c(sched$home_team, sched$away_team))))
    return(all_teams %>%
             mutate(
               games_played = 0,
               win_pct = 0, home_win_pct = 0, away_win_pct = 0,
               last_game_day = as.Date(NA),
               epa_per_play = 0, qb_epa = 0, third_down_pct = 0,
               turnovers_per_game = 0, scoring_plays_per_game = 0
             ))
  }
  build_team_features_through(sched, pbp, through_week = wk)
}
