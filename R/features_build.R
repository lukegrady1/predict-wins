# R/features_build.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(janitor)
})

# =============================================================================
# STEP 1: Team-week outcomes + per-game PBP stats (leak-safe)
# =============================================================================

# Final score outcomes per game -> expand to (team, opponent)
summarize_team_game_outcomes <- function(pbp) {
  pbp <- pbp %>%
    filter(!is.na(game_id), !is.na(week), !is.na(season)) %>%
    clean_names() %>%
    mutate(
      # optional flags you might extend later
      is_pass = coalesce(pass == 1, FALSE),
      is_rush = coalesce(rush == 1, FALSE),
      is_third_down_att  = (down == 3L) & (is_pass | is_rush),
      is_third_down_conv = coalesce(third_down_converted, FALSE) & is_third_down_att,
      # turnovers = interception OR fumble lost
      is_turnover = coalesce(interception == 1, FALSE) | coalesce(fumble_lost == 1, FALSE)
    )

  game_final <- pbp %>%
    group_by(game_id, season, week, home_team, away_team) %>%
    summarise(
      home_score_final = last(home_score),
      away_score_final = last(away_score),
      .groups = "drop"
    )

  home_rows <- game_final %>%
    transmute(
      season, week, game_id,
      team     = home_team,
      opponent = away_team,
      points_for     = home_score_final,
      points_against = away_score_final
    )

  away_rows <- game_final %>%
    transmute(
      season, week, game_id,
      team     = away_team,
      opponent = home_team,
      points_for     = away_score_final,
      points_against = home_score_final
    )

  bind_rows(home_rows, away_rows) %>%
    mutate(
      point_diff = points_for - points_against,
      win        = as.integer(point_diff > 0)
    ) %>%
    arrange(season, week, game_id, team)
}

# Per-team per-game PBP stats: ypp, success_rate, run/pass splits
compute_team_game_stats <- function(pbp) {
  pbp %>%
    clean_names() %>%
    filter(!is.na(game_id), !is.na(week), !is.na(season)) %>%
    # keep only actual run/pass plays for team-level rate stats
    mutate(is_play = coalesce(rush == 1, FALSE) | coalesce(pass == 1, FALSE)) %>%
    filter(is_play, !is.na(posteam)) %>%
    group_by(season, week, game_id, team = posteam) %>%
    summarise(
      total_plays   = n(),
      total_yards   = sum(coalesce(yards_gained, 0), na.rm = TRUE),
      yards_per_play = if_else(total_plays > 0, total_yards / total_plays, NA_real_),
      success_rate  = mean(coalesce(epa, NA_real_) > 0, na.rm = TRUE),
      pass_attempts = sum(coalesce(pass == 1, FALSE)),
      rush_attempts = sum(coalesce(rush == 1, FALSE)),
      pass_pct      = if_else(total_plays > 0, pass_attempts / total_plays, NA_real_),
      rush_pct      = if_else(total_plays > 0, rush_attempts / total_plays, NA_real_),
      .groups = "drop"
    )
}

# Merge outcomes + per-game stats -> base team-week frame
build_base_team_week <- function(pbp) {
  outcomes <- summarize_team_game_outcomes(pbp)
  stats    <- compute_team_game_stats(pbp)

  outcomes %>%
    left_join(stats, by = c("season", "week", "game_id", "team")) %>%
    # If a team has zero qualifying plays (rare), fill zeros where safe
    mutate(
      total_plays   = coalesce(total_plays, 0L),
      total_yards   = coalesce(total_yards, 0),
      yards_per_play = if_else(total_plays > 0, yards_per_play, NA_real_),
      success_rate  = success_rate,  # leave NA if no plays
      pass_attempts = coalesce(pass_attempts, 0L),
      rush_attempts = coalesce(rush_attempts, 0L),
      pass_pct      = if_else(total_plays > 0, pass_attempts / total_plays, NA_real_),
      rush_pct      = if_else(total_plays > 0, rush_attempts / total_plays, NA_real_)
    )
}

# =============================================================================
# STEP 2: Strength of Schedule (opponent pre-week strength)
# =============================================================================

# Expand schedules to team-week rows with opponent
build_team_week_map <- function(sched) {
  sched %>%
    clean_names() %>%
    filter(!is.na(game_id), !is.na(week), !is.na(season)) %>%
    {
      home_rows <- transmute(.,
        season, week, game_id,
        team     = home_team,
        opponent = away_team
      )
      away_rows <- transmute(.,
        season, week, game_id,
        team     = away_team,
        opponent = home_team
      )
      bind_rows(home_rows, away_rows)
    }
}

# Compute opponent pre-week win% / avg point diff from team-game summaries
compute_team_strength_preweek <- function(team_game_summaries) {
  team_game_summaries %>%
    mutate(week = as.integer(week)) %>%
    arrange(season, team, week) %>%
    group_by(season, team) %>%
    mutate(
      win_num      = as.integer(win == 1),
      gp_to_date   = lag(row_number()),
      wins_cum_pre = lag(cumsum(win_num)),
      pd_cum_pre   = lag(cumsum(point_diff)),
      win_pct_pre  = if_else(gp_to_date > 0, wins_cum_pre / gp_to_date, NA_real_),
      pd_avg_pre   = if_else(gp_to_date > 0, pd_cum_pre / gp_to_date, NA_real_)
    ) %>%
    ungroup() %>%
    select(season, week, team, gp_to_date, win_pct_pre, pd_avg_pre)
}

build_sos_teamweek <- function(sched, team_game_summaries) {
  team_week_map <- build_team_week_map(sched)

  opp_strength_pre <- compute_team_strength_preweek(team_game_summaries) %>%
    rename(
      opponent        = team,
      opp_gp_pre      = gp_to_date,
      opp_win_pct_pre = win_pct_pre,
      opp_pd_avg_pre  = pd_avg_pre
    )

  team_week_map %>%
    left_join(opp_strength_pre, by = c("season", "week", "opponent"))
}

attach_sos_to_features <- function(features_tw, sched, team_game_summaries) {
  sos_tw <- build_sos_teamweek(sched, team_game_summaries) %>%
    select(season, week, team, opp_gp_pre, opp_win_pct_pre, opp_pd_avg_pre)

  features_tw %>%
    left_join(sos_tw, by = c("season", "week", "team"))
}

# =============================================================================
# STEP 3: Quarterback features (starter + change vs previous week)
# =============================================================================

build_qb_teamweek <- function(sched) {
  sched %>%
    clean_names() %>%
    filter(!is.na(week), !is.na(season)) %>%
    {
      home_rows <- transmute(.,
        season, week, game_id,
        team     = home_team,
        qb_id    = home_qb_id,
        qb_name  = home_qb_name
      )
      away_rows <- transmute(.,
        season, week, game_id,
        team     = away_team,
        qb_id    = away_qb_id,
        qb_name  = away_qb_name
      )
      bind_rows(home_rows, away_rows)
    } %>%
    arrange(season, team, week) %>%
    group_by(season, team) %>%
    mutate(
      qb_prev_id = lag(qb_id),
      qb_change  = as.integer(!is.na(qb_id) & !is.na(qb_prev_id) & qb_id != qb_prev_id)
    ) %>%
    # first appearance has no prior QB; treat as no change (0)
    mutate(qb_change = replace_na(qb_change, 0L)) %>%
    ungroup() %>%
    select(season, week, team, qb_id, qb_name, qb_change)
}

attach_qb_to_features <- function(features_tw, sched) {
  qb_tw <- build_qb_teamweek(sched)
  features_tw %>%
    left_join(qb_tw, by = c("season", "week", "team"))
}

# =============================================================================
# Orchestrators used by scripts/02_build_features.R
# =============================================================================

build_features_team_week <- function(pbp, sched) {
  base_team_week <- build_base_team_week(pbp)

  base_team_week %>%
    # Step 2: SoS uses opponent pre-week strength computed from base outcomes
    attach_sos_to_features(sched = sched, team_game_summaries = base_team_week) %>%
    # Step 3: QB features
    attach_qb_to_features(sched = sched) %>%
    arrange(season, week, team)
}

# Returns REG-season features for one season (as expected by 02_build_features.R)
build_training_features_one_season <- function(sched, pbp) {
  sched_reg <- sched %>% filter(game_type == "REG")
  feats <- build_features_team_week(pbp, sched_reg)

  reg_keys <- build_team_week_map(sched_reg) %>%
    select(season, week, team) %>%
    distinct()

  feats %>%
    inner_join(reg_keys, by = c("season", "week", "team"))
}

# Current-season snapshot through week_cutoff - 1 (REG only)
build_current_features_through_week <- function(sched, pbp, week_cutoff) {
  sched_reg <- sched %>% filter(game_type == "REG")

  if (!is.numeric(week_cutoff) || week_cutoff <= 1 || nrow(pbp) == 0) {
    return(tibble(
      season          = integer(),
      week            = integer(),
      game_id         = character(),
      team            = character(),
      opponent        = character(),
      win             = integer(),
      point_diff      = numeric(),
      total_plays     = integer(),
      total_yards     = numeric(),
      yards_per_play  = numeric(),
      success_rate    = numeric(),
      pass_attempts   = integer(),
      rush_attempts   = integer(),
      pass_pct        = numeric(),
      rush_pct        = numeric(),
      opp_gp_pre      = numeric(),
      opp_win_pct_pre = numeric(),
      opp_pd_avg_pre  = numeric(),
      qb_id           = character(),
      qb_name         = character(),
      qb_change       = integer()
    ))
  }

  build_features_team_week(pbp, sched_reg) %>%
    filter(week < week_cutoff)
}
