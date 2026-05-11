# Borrower segmentation using origination-time credit risk features

library(dplyr)
library(readr)
library(tidyr)
library(cluster)

DATA_PATH <- "data/early_2012_2013_loan_sample_with_outcome.csv"

df <- read_csv(DATA_PATH, show_col_types = FALSE)

df <- df %>%
  mutate(
    revol_util = as.numeric(gsub("%", "", revol_util)),
    grade = factor(grade, levels = c("A", "B", "C", "D", "E", "F", "G")),
    grade_num = as.numeric(grade),
    Approval_A = grade_num < 1.5 & dti <= 10 & annual_inc > 80000,
    Approval_B = grade_num < 3.5 & dti <= 30 & annual_inc > 40000
  )

cluster_vars <- c(
  "annual_inc",
  "dti",
  "loan_amnt",
  "int_rate",
  "delinq_2yrs",
  "inq_last_6mths",
  "open_acc",
  "revol_util",
  "total_acc",
  "revol_bal"
)

df_clust <- df %>%
  select(all_of(cluster_vars), loan_is_bad, grade, Approval_A, Approval_B, id) %>%
  tidyr::drop_na()

clust <- df_clust %>% select(all_of(cluster_vars))

maha <- mahalanobis(clust, colMeans(clust), cov(clust))
maha_p_value <- pchisq(maha, df = ncol(clust) - 1, lower.tail = FALSE)

df_clust <- df_clust[maha_p_value >= 0.001, ]
clust <- clust[maha_p_value >= 0.001, ]

clust_scaled <- scale(clust)

set.seed(42)
sample_index <- sample(nrow(clust_scaled), 5000)
clust_sample <- clust_scaled[sample_index, ]

distance_mat <- dist(clust_sample, method = "euclidean")
hierarchical_model <- hclust(distance_mat, method = "ward.D2")

silhouette_scores <- sapply(2:10, function(k) {
  cluster_assignment <- cutree(hierarchical_model, k = k)
  mean(silhouette(cluster_assignment, distance_mat)[, 3])
})

print(data.frame(k = 2:10, silhouette = silhouette_scores))

set.seed(42)
kmeans_model <- kmeans(clust_scaled, centers = 3, nstart = 25)

df_clust$cluster <- kmeans_model$cluster

profile_table <- df_clust %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    bad_loan_rate = round(mean(loan_is_bad), 4),
    int_rate = round(mean(int_rate), 2),
    dti = round(mean(dti), 2),
    annual_inc = round(mean(annual_inc), 0),
    revol_util = round(mean(revol_util), 1),
    loan_amnt = round(mean(loan_amnt), 0),
    .groups = "drop"
  )

print(profile_table)
