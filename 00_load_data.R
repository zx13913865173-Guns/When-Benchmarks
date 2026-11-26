# 00_load_data.R — assemble the modelling data frame (paper section 2.8).
# Run from within 06_statistics/.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})

# --- paths ------------------------------------------------------------------
META_CSV   <- "../03_benchmark/benchmark_metadata.csv"
PRED_CSV   <- "../outputs/pred_labels.csv"
TRUTH_CSV  <- "../07_annotation/ground_truth.csv"
OUT_RDS    <- "../outputs/model_data.rds"

# --- load -------------------------------------------------------------------
meta  <- read_csv(META_CSV,  show_col_types = FALSE)
pred  <- read_csv(PRED_CSV,  show_col_types = FALSE)
truth <- read_csv(TRUTH_CSV, show_col_types = FALSE)

# --- per-image long table: one row per (image, style-slot) -------------------
# style-slot is "dominant"/"secondary" for asymmetric images and "pos1"/"pos2"
# for equal-weight images. Pure-style controls are excluded from these contrasts.
long <- meta %>%
  filter(category != "pure_style_control") %>%
  mutate(
    style_a_origin = recode(style_a, "ink_wash" = "East", "ukiyo_e" = "East",
                            "cyberpunk" = "West", "baroque" = "West",
                            "minimalist" = "None"),
    style_b_origin = recode(style_b, "ink_wash" = "East", "ukiyo_e" = "East",
                            "cyberpunk" = "West", "baroque" = "West",
                            "minimalist" = "None")
  )

# --- join predictions and ground truth --------------------------------------
stopifnot(all(c("image_id", "pred_styles") %in% names(pred)))
model_data <- long %>%
  left_join(pred,  by = "image_id") %>%
  left_join(truth, by = "image_id")

saveRDS(model_data, OUT_RDS)
message("model_data rows: ", nrow(model_data), " -> ", OUT_RDS)
