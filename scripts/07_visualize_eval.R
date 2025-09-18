# scripts/06_visualize_eval.R
# Purpose: Visualize model evaluation results produced by 05_eval_model.R
# Inputs (from data_processed/):
#   - metrics_training.csv
#   - predictions_training.parquet
# Outputs (to reports/):
#   - auc_by_season.png
#   - roc_overall.png
#   - pr_overall.png
#   - calibration_curve.png
#   - prob_density.png
#   - (optional) gain_curve.png, lift_curve.png
#   - eval_report.html  (a simple HTML that embeds all PNGs)

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(arrow)
  library(glue)
  library(fs)
  library(tidymodels)  # yardstick for curves
})

source("R/utils_io.R")
paths <- read_paths()

# --- Setup --------------------------------------------------------------------
set.seed(42)
fs::dir_create("reports")

metrics_path <- fs::path(paths$data_processed, "metrics_training.csv")
preds_path   <- fs::path(paths$data_processed, "predictions_training.parquet")

if (!fs::file_exists(metrics_path) || !fs::file_exists(preds_path)) {
  stop("Missing metrics or predictions. Run scripts/05_eval_model.R first.")
}

metrics <- readr::read_csv(metrics_path, show_col_types = FALSE) %>% clean_names()
preds   <- arrow::read_parquet(preds_path) %>% clean_names()

# Ensure expected column names
if (!"p_home_win" %in% names(preds)) {
  if (".pred_yes" %in% names(preds)) preds <- preds %>% rename(p_home_win = .pred_yes)
}
# yardstick expects factor truth with positive class explicitly set
preds <- preds %>% mutate(home_win = factor(home_win, levels = c("no","yes")))

# --- 1) AUC by season (bar chart) ---------------------------------------------
metrics_clean <- metrics %>%
  mutate(season_label = if_else(is.na(season), "OVERALL", as.character(season)))

p_auc <- metrics_clean %>%
  filter(season_label != "OVERALL") %>%
  ggplot(aes(x = as.factor(season_label), y = roc_auc)) +
  geom_col() +
  geom_hline(data = metrics_clean %>% filter(season_label == "OVERALL"),
             aes(yintercept = roc_auc), linetype = "dashed") +
  labs(title = "ROC AUC by Season (dashed = OVERALL)",
       x = "Season", y = "ROC AUC") +
  coord_cartesian(ylim = c(0.5, 1.0)) +
  theme_minimal(base_size = 12)

auc_png <- "reports/auc_by_season.png"
ggsave(auc_png, p_auc, width = 9, height = 4.5, dpi = 150)

# --- 2) Overall ROC curve -----------------------------------------------------
roc_df  <- yardstick::roc_curve(preds, truth = home_win, p_home_win, event_level = "second")
roc_auc <- yardstick::roc_auc(preds, truth = home_win, p_home_win, event_level = "second")$.estimate[1]

p_roc <- roc_df %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(size = 1) +
  geom_abline(linetype = "dashed") +
  annotate("text", x = 0.7, y = 0.1, label = glue("AUC = {round(roc_auc, 3)}")) +
  labs(title = "ROC Curve (Overall)", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 12)

roc_png <- "reports/roc_overall.png"
ggsave(roc_png, p_roc, width = 6, height = 5, dpi = 150)

# --- 3) Precision–Recall curve ------------------------------------------------
pr_df    <- yardstick::pr_curve(preds, truth = home_win, p_home_win, event_level = "second")
avg_prec <- yardstick::average_precision(preds, truth = home_win, p_home_win, event_level = "second")$.estimate[1]

p_pr <- pr_df %>%
  ggplot(aes(x = recall, y = precision)) +
  geom_path(size = 1) +
  annotate("text", x = 0.1, y = 0.95, label = glue("AP = {round(avg_prec, 3)}")) +
  labs(title = "Precision–Recall Curve (Overall)", x = "Recall", y = "Precision") +
  theme_minimal(base_size = 12)

pr_png <- "reports/pr_overall.png"
ggsave(pr_png, p_pr, width = 6, height = 5, dpi = 150)

# --- 4) Calibration (reliability) curve --------------------------------------
calib_bins <- 20
calib_df <- preds %>%
  transmute(p = p_home_win, y = as.integer(home_win == "yes")) %>%
  mutate(bin = cut(p, breaks = seq(0, 1, length.out = calib_bins + 1), include.lowest = TRUE, right = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    n = n(),
    p_mean = mean(p, na.rm = TRUE),
    y_rate = mean(y, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n > 0)

p_cal <- calib_df %>%
  ggplot(aes(x = p_mean, y = y_rate, size = n)) +
  geom_point(alpha = 0.8) +
  geom_abline(linetype = "dashed") +
  labs(title = "Calibration (Reliability) Curve",
       x = "Predicted probability (bin mean)",
       y = "Observed event rate",
       size = "Bin N") +
  coord_equal(xlim = c(0,1), ylim = c(0,1)) +
  theme_minimal(base_size = 12)

cal_png <- "reports/calibration_curve.png"
ggsave(cal_png, p_cal, width = 6, height = 5, dpi = 150)

# Save a calibration table for quick inspection
readr::write_csv(calib_df, "reports/calibration_table.csv")

# --- 5) Probability density plot by class ------------------------------------
p_density <- preds %>%
  ggplot(aes(x = p_home_win, fill = home_win)) +
  geom_density(alpha = 0.4, adjust = 1.2) +
  labs(title = "Predicted Probability Density by Outcome",
       x = "p(home win)", y = "Density", fill = "Truth") +
  theme_minimal(base_size = 12)

dens_png <- "reports/prob_density.png"
ggsave(dens_png, p_density, width = 7, height = 4.5, dpi = 150)

# --- 6) (Optional) Gain & Lift curves ----------------------------------------
gains_df <- preds %>%
  transmute(p = p_home_win, y = as.integer(home_win == "yes")) %>%
  arrange(desc(p)) %>%
  mutate(row = row_number(),
         cum_positives = cumsum(y),
         total_positives = sum(y),
         gain = cum_positives / total_positives,
         pct_of_data = row / n()) %>%
  select(pct_of_data, gain)

p_gain <- gains_df %>%
  ggplot(aes(x = pct_of_data, y = gain)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed") +
  labs(title = "Cumulative Gains Curve", x = "% of Samples (sorted by p)", y = "% of Positives Captured") +
  theme_minimal(base_size = 12)

gain_png <- "reports/gain_curve.png"
ggsave(gain_png, p_gain, width = 6, height = 5, dpi = 150)

p_lift <- gains_df %>%
  mutate(lift = gain / pct_of_data) %>%
  ggplot(aes(x = pct_of_data, y = lift)) +
  geom_line(size = 1) +
  labs(title = "Lift Curve", x = "% of Samples", y = "Lift") +
  theme_minimal(base_size = 12)

lift_png <- "reports/lift_curve.png"
ggsave(lift_png, p_lift, width = 6, height = 5, dpi = 150)

# --- 7) Simple HTML report ----------------------------------------------------
html_path <- "reports/eval_report.html"
html <- glue::glue(
  '<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Model Evaluation</title>
<style>body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;}}
h1,h2{{margin:0 0 8px}} .card{{margin-bottom:24px}}</style></head>
<body>
<h1>Model Evaluation</h1>
<div class="card"><h2>ROC AUC by Season</h2><img src="auc_by_season.png" width="900"></div>
<div class="card"><h2>Overall ROC Curve</h2><img src="roc_overall.png" width="650"></div>
<div class="card"><h2>Precision–Recall Curve</h2><img src="pr_overall.png" width="650"></div>
<div class="card"><h2>Calibration Curve</h2><img src="calibration_curve.png" width="650"></div>
<div class="card"><h2>Probability Density by Outcome</h2><img src="prob_density.png" width="800"></div>
<div class="card"><h2>Cumulative Gains</h2><img src="gain_curve.png" width="650"></div>
<div class="card"><h2>Lift Curve</h2><img src="lift_curve.png" width="650"></div>
<p><a href="calibration_table.csv">Download calibration table (CSV)</a></p>
</body></html>'
)
writeLines(html, con = html_path)

message(glue("Wrote plots and report to: {fs::path_abs('reports')}"))
