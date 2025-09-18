# R/modeling.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(jsonlite)
})

# Treat "win" as the positive class when levels = c("loss","win")
options(yardstick.event_level = "second")

# ---- Coerce outcome to factor(loss, win) ----
coerce_outcome_factor <- function(df, outcome = "win") {
  x <- df[[outcome]]
  if (is.factor(x)) {
    # Relevel to c("loss","win") if different
    lev <- levels(x)
    if (!identical(lev, c("loss","win"))) {
      if (all(c("loss","win") %in% lev)) df[[outcome]] <- factor(as.character(x), levels = c("loss","win"))
      else {
        # attempt smart map
        y <- ifelse(as.character(x) %in% c("W","w","win","Win","1","TRUE","true"), "win", "loss")
        df[[outcome]] <- factor(y, levels = c("loss","win"))
      }
    }
    return(df)
  }
  
  if (is.logical(x)) {
    df[[outcome]] <- factor(ifelse(x, "win", "loss"), levels = c("loss","win"))
    return(df)
  }
  
  if (is.numeric(x)) {
    df[[outcome]] <- factor(ifelse(x == 1, "win", "loss"), levels = c("loss","win"))
    return(df)
  }
  
  if (is.character(x)) {
    df[[outcome]] <- case_when(
      x %in% c("W","w","win","Win","WIN","1","TRUE","true") ~ "win",
      TRUE ~ "loss"
    ) |> factor(levels = c("loss","win"))
    return(df)
  }
  
  stop("Unsupported outcome type; please convert to factor(loss, win).")
}

# ---- Pick a formula based on available columns ----
# We build from common, robust predictors:
# - performance: epa/pp, qb_epa, third down, scoring plays, turnovers
# - team strength: win pct (overall, home, away)
# - rest: rest days for each side
# Works with either home_/away_ *or* team/opp_* naming
pick_formula <- function(df, outcome = "win") {
  cols <- names(df)
  
  # Candidate pairs: (A vs B) must both be present
  pairs <- list(
    c("home_epa_per_play", "away_epa_per_play"),
    c("home_qb_epa", "away_qb_epa"),
    c("home_third_down_pct", "away_third_down_pct"),
    c("home_scoring_plays_per_game", "away_scoring_plays_per_game"),
    c("home_turnovers_per_game", "away_turnovers_per_game"),
    c("home_win_pct", "away_win_pct"),
    c("home_home_win_pct", "away_home_win_pct"),
    c("home_away_win_pct", "away_away_win_pct"),
    c("home_rest_days", "away_rest_days"),
    # Optional Elo if present
    c("home_elo", "away_elo")
  )
  
  terms <- c()
  for (p in pairs) if (all(p %in% cols)) terms <- c(terms, p)
  
  # If no home/away, try team/opp_* scheme
  if (length(terms) == 0) {
    pairs2 <- list(
      c("epa_per_play","opp_epa_per_play"),
      c("qb_epa","opp_qb_epa"),
      c("third_down_pct","opp_third_down_pct"),
      c("scoring_plays_per_game","opp_scoring_plays_per_game"),
      c("turnovers_per_game","opp_turnovers_per_game"),
      c("win_pct","opp_win_pct"),
      c("home_win_pct","opp_home_win_pct"),
      c("away_win_pct","opp_away_win_pct"),
      c("rest_days","opp_rest_days"),
      c("elo","opp_elo")
    )
    for (p in pairs2) if (all(p %in% cols)) terms <- c(terms, p)
  }
  
  if (length(terms) == 0) stop("No usable predictor pairs found in model_frame. Check your columns.")
  
  rhs <- paste(unique(terms), collapse = " + ")
  as.formula(paste(outcome, "~", rhs))
}

# ---- Create recipe/workflow ----
make_recipe <- function(formula, df, outcome = "win") {
  recipe(formula, data = df) %>%
    step_zv(all_predictors()) %>%
    step_impute_median(all_numeric_predictors()) %>%   # light imputation
    step_normalize(all_numeric_predictors())
}

make_workflow <- function(rec) {
  lr_spec <- logistic_reg(penalty = 0, mixture = 1) %>% set_engine("glm")
  workflow() %>% add_model(lr_spec) %>% add_recipe(rec)
}

# ---- Fit / evaluate ----
fit_and_eval <- function(df, outcome = "win", test_prop = 0.2, seed = 123) {
  set.seed(seed)
  split <- initial_split(df, prop = 1 - test_prop, strata = !!sym(outcome))
  train <- training(split)
  test  <- testing(split)
  
  form <- pick_formula(train, outcome = outcome)
  rec  <- make_recipe(form, train, outcome = outcome)
  wf   <- make_workflow(rec)
  fit  <- fit(wf, data = train)
  
  preds <- bind_cols(
    test %>% select(all_of(outcome)),
    predict(fit, test, type = "class"),
    predict(fit, test, type = "prob")
  )
  
  mets <- list(
    accuracy = yardstick::accuracy(preds, truth = !!sym(outcome), estimate = .pred_class) %>% as_tibble(),
    roc_auc  = if (".pred_win" %in% names(preds)) yardstick::roc_auc(preds, truth = !!sym(outcome), .pred_win) %>% as_tibble() else tibble(.metric="roc_auc", .estimator="binary", .estimate=NA_real_)
  )
  
  list(fit = fit, formula = deparse(form), metrics = mets)
}

# ---- Save artifacts ----
save_artifacts <- function(fit, formula, metrics, feature_cols, models_dir = "models") {
  if (!dir.exists(models_dir)) dir.create(models_dir, recursive = TRUE)
  
  saveRDS(fit, file.path(models_dir, "fit_lr.rds"))
  
  # feature map helps future scripts build matching columns for prediction
  feature_info <- list(
    formula = formula,
    features = feature_cols
  )
  write_json(feature_info, file.path(models_dir, "feature_info.json"), auto_unbox = TRUE, pretty = TRUE)
  
  # flatten metrics for a quick glance
  flat <- bind_rows(
    metrics$accuracy %>% mutate(type = "accuracy"),
    metrics$roc_auc  %>% mutate(type = "roc_auc")
  )
  write_json(flat, file.path(models_dir, "metrics.json"), auto_unbox = TRUE, pretty = TRUE)
}
