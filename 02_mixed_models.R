# 02_mixed_models.R — condition-level mixed-effects models (paper section 2.8).
# Binomial GLMM with image-level random intercept, fitted with lme4 1.1-35 and
# p-values from lmerTest (Satterthwaite approximation).

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lme4)
  library(lmerTest)
})

model_data <- readRDS("../outputs/model_data.rds")

# --- build the per-(image, style-slot) binary response -----------------------
# For each image we test whether the interrogator recalled the dominant vs the
# secondary style. Only the 60 asymmetric images enter this contrast.
asym <- model_data %>%
  filter(category == "asymmetric_primary" | category == "asymmetric_generalization") %>%
  mutate(
    hit_dominant  = as.integer(grepl(dominant_style,  pred_styles, fixed = TRUE)),
    hit_secondary = as.integer(grepl(secondary_style, pred_styles, fixed = TRUE)),
    style_side    = factor(c("dominant", "secondary"), levels = c("dominant", "secondary"))
  )

# long form: two rows per image (dominant / secondary)
long <- bind_rows(
  asym %>% transmute(image_id, pair, seed, style_side = "dominant",
                     hit = hit_dominant, origin = "mixed"),
  asym %>% transmute(image_id, pair, seed, style_side = "secondary",
                     hit = hit_secondary, origin = "mixed")
) %>%
  mutate(style_side = factor(style_side, levels = c("dominant", "secondary")))

# --- GLMM: does the dominant side dominate the secondary side? ---------------
m1 <- glmer(hit ~ style_side + (1 | image_id) + (1 | pair),
            data = long, family = binomial)

cat("=== Model 1: hit ~ style_side + (1|image) + (1|pair) ===\n")
print(summary(m1)$coefficients)

# --- GLMM: East-West asymmetry as secondary (the key contrast) ---------------
# Restrict to the primary pairs and code the secondary side's origin.
primary_pairs <- c("ink_cyber", "ukiyo_baroque", "cyber_baroque", "ink_baroque")
east_west <- long %>%
  filter(pair %in% primary_pairs, style_side == "secondary") %>%
  mutate(origin = ifelse(pair %in% c("ink_cyber", "ukiyo_baroque", "ink_baroque"), "East", "West"))

m2 <- glmer(hit ~ origin + (1 | image_id) + (1 | pair),
            data = east_west, family = binomial)

cat("\n=== Model 2: secondary hit ~ origin + (1|image) + (1|pair) ===\n")
print(summary(m2)$coefficients)

saveRDS(list(m1 = m1, m2 = m2), "../outputs/mixed_models.rds")
message("Saved mixed_models.rds")
