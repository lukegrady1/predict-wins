# R/evaluate.R
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
})

options(yardstick.event_level = "second") # positive = "win"

# Return accuracy, ROC AUC, confusion matrix, calibration tibble
evaluate_holdout <- function(fit, df, outcome = "win", test_prop = 0.2, seed = 123) {
  set.seed(seed)
  split <- initial_split(df, prop = 1 - test_prop, strata = !!sym(outcome))
  train <- training(split)
  test  <- testing(split)
  
  preds <- bind_cols(
    test %>% select(all_of(outcome)),
    predict(fit, test, type = "class"),
    predict(fit, test, type = "prob")
  )
  
  acc <- accuracy(preds, truth = !!sym(outcome), estimate = .pred_class)
  auc <- if (".pred_win" %in% names(preds)) roc_auc(preds, truth = !!sym(outcome), .pred_win) else
    tibble(.metric="roc_auc", .estimator="binary", .estimate=NA_real_)
  
  cm  <- yardstick::conf_mat(preds, truth = !!sym(outcome), estimate = .pred_class)
  
  # simple calibration by 10 bins
  calib <- preds %>%
    mutate(bin = ntile(.pred_win, 10L)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(.pred_win, na.rm = TRUE),
      emp_win   = mean(!!sym(outcome) == "win", na.rm = TRUE),
      n = n(), .groups = "drop"
    )
  
  list(metrics = list(accuracy = acc, roc_auc = auc),
       conf_mat = cm, calibration = calib, preds = preds)
}
