# Loan approval policy A/B testing

library(dplyr)
library(readr)

DATA_PATH <- "data/early_2012_2013_loan_sample_with_outcome.csv"

df <- read_csv(DATA_PATH, show_col_types = FALSE)

df <- df %>%
  mutate(
    grade = factor(grade, levels = c("A", "B", "C", "D", "E", "F", "G")),
    grade_num = as.numeric(grade),
    Approval_A = grade_num < 1.5 & dti <= 10 & annual_inc > 80000,
    Approval_B = grade_num < 3.5 & dti <= 30 & annual_inc > 40000
  ) %>%
  filter(!is.na(dti), !is.na(annual_inc), !is.na(loan_is_bad))

policy_metrics <- function(data, approval_col) {
  approved <- data[[approval_col]]
  bad <- data$loan_is_bad == TRUE

  false_negative <- sum(approved & bad, na.rm = TRUE)
  true_positive <- sum(!approved & bad, na.rm = TRUE)
  true_negative <- sum(approved & !bad, na.rm = TRUE)
  false_positive <- sum(!approved & !bad, na.rm = TRUE)

  tibble(
    policy = approval_col,
    n = nrow(data),
    approval_rate = mean(approved, na.rm = TRUE),
    false_negative_rate = false_negative / (true_positive + false_negative),
    default_rate_if_approved = false_negative / sum(approved, na.rm = TRUE),
    false_negative = false_negative,
    true_positive = true_positive,
    true_negative = true_negative,
    false_positive = false_positive
  )
}

results <- bind_rows(
  policy_metrics(df, "Approval_A"),
  policy_metrics(df, "Approval_B")
)

print(results)

bad_loans <- df %>% filter(loan_is_bad == TRUE)

test_result <- prop.test(
  x = c(sum(bad_loans$Approval_A), sum(bad_loans$Approval_B)),
  n = c(nrow(bad_loans), nrow(bad_loans)),
  alternative = "less",
  correct = FALSE
)

print(test_result)
