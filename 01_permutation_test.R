# 01_permutation_test.R — exact pair-level permutation test (paper section 2.8).
#
# Design: order the 6 fusion pairs by secondary-style recall, then read off the
# East/West origin of the dominant-on-secondary side. The test statistic is the
# number of ADJACENT SAME-ORIGIN pairs (East-East or West-West):
#   T = 0  -> perfectly alternating origins
#   T = 5  -> fully segregated origins
# The observed arrangement is closer to segregated than chance, so the one-tailed
# p-value is the LOWER tail, P(T <= T_obs), over all 6! = 720 permutations.
# By symmetry the minimum attainable one-tailed p-value is 1/360 ~= 0.0028.

suppressPackageStartupMessages({ library(dplyr); library(readr) })

# --- input: pairs ordered by secondary recall (from metrics.py output) --------
# Provide a CSV with columns: pair, secondary_recall, origin_of_secondary_side
# origin_of_secondary_side in {"East","West"} — the origin recalled second-most.
pair_file <- "../outputs/pairs_secondary_recall.csv"
if (!file.exists(pair_file)) {
  stop("Provide ", pair_file, " with pair-level secondary recall first.")
}
pairs <- read_csv(pair_file, show_col_types = FALSE) %>% arrange(secondary_recall)

origins <- pairs$origin_of_secondary_side
n <- length(origins)
stopifnot(n == 6)

# --- statistic ---------------------------------------------------------------
n_adjacent_same <- function(x) sum(x[-length(x)] == x[-1])

T_obs <- n_adjacent_same(origins)

# --- exact enumeration of all permutations ----------------------------------
perms <- function(x) {
  if (length(x) == 1) return(list(x))
  unlist(lapply(seq_along(x), function(i)
    lapply(perms(x[-i]), function(p) c(x[i], p))), recursive = FALSE)
}
all_perms <- perms(origins)
T_perm <- vapply(all_perms, n_adjacent_same, integer(1))

p_lower <- mean(T_perm <= T_obs)
p_min   <- 1 / 360

cat(sprintf("T_obs (adjacent same-origin) = %d\n", T_obs))
cat(sprintf("n permutations              = %d (6! = 720)\n", length(T_perm)))
cat(sprintf("one-tailed lower p          = %.4f\n", p_lower))
cat(sprintf("minimum attainable p        = %.4f (1/360)\n", p_min))

write_csv(data.frame(T_obs = T_obs, p_lower = p_lower, p_min = p_min, n_perm = length(T_perm)),
          "../outputs/permutation_test.csv")
