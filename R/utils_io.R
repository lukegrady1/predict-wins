# R/utils_io.R
suppressPackageStartupMessages({
  library(fs)
  library(arrow)
  library(glue)
  library(yaml)
})

read_paths <- function(path = "config/paths.yml") {
  yaml::read_yaml(path)
}

read_params <- function(path = "config/params.yml") {
  yaml::read_yaml(path)
}

ensure_dirs <- function(...) {
  dirs <- c(...)
  purrr::walk(dirs, ~ if (!fs::dir_exists(.x)) fs::dir_create(.x, recurse = TRUE))
}

# Parquet helpers -------------------------------------------------------------

write_parquet_safe <- function(df, path) {
  dir <- fs::path_dir(path)
  if (!fs::dir_exists(dir)) fs::dir_create(dir, recurse = TRUE)
  arrow::write_parquet(df, path)
  invisible(path)
}

read_parquet_safe <- function(path) {
  stopifnot(fs::file_exists(path))
  arrow::read_parquet(path)
}

# CSV helpers (handy for quick looks) ----------------------------------------

write_csv_safe <- function(df, path) {
  dir <- fs::path_dir(path)
  if (!fs::dir_exists(dir)) fs::dir_create(dir, recurse = TRUE)
  readr::write_csv(df, path)
  invisible(path)
}
