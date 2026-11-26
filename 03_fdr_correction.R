# 03_fdr_correction.R — Benjamini-Hochberg FDR correction across all recall contrasts.
# Paper section 2.8 applies BH across every recall comparison reported.

suppressPackageStartupMessages({ library(dplyr); library(readr) })

# Collect p-values produced by the model scripts.
# Expected long CSV: contrast, estimate, std_error, p_value
pfile <- "../outputs/recall_contrasts.csv"
stopifnot(file.exists(pfile))
contrasts <- read_csv(pfile, show_col_types = FALSE)

contrasts <- contrasts %>%
  mutate(p_adj_BH = p.adjust(p_value, method = "BH"),
         significant = p_adj_BH < 0.05) %>%
  arrange(p_value)

print(contrasts, n = Inf)

write_csv(contrasts, "../outputs/recall_contrasts_fdr.csv")
message("BH-adjusted p-values written to ../outputs/recall_contrasts_fdr.csv")
