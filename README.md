# Loan Default Risk Analytics
[中文版本](README_zh.md) | English

Credit risk analytics project using Lending Club loan data to compare approval policies, segment borrowers, and prototype default prediction models.

## Project Overview

This project analyses consumer loan applications from a credit-risk decisioning perspective. The workflow combines rule-based policy evaluation, borrower segmentation, and supervised default prediction to support approval, rejection, and manual-review decisions.

## Business Questions

1. Which loan approval policy better reduces the risk of approving bad loans?
2. Can borrowers be segmented into interpretable risk groups using observable loan-origination features?
3. Does a deep learning model improve default prediction compared with a simpler logistic regression baseline?
4. How can model scores support a practical three-way review policy?

## Methods

- Policy A/B testing using false negative rate as the key risk metric
- Two-proportion hypothesis testing to compare policy performance
- Borrower segmentation using hierarchical clustering and k-means
- Feature engineering for default prediction, including credit history, income burden, utilisation, and missingness indicators
- PyTorch DNN model with class imbalance handling
- Logistic regression baseline for model comparison
- Threshold tuning and cluster-level model evaluation

## Key Results

- Analysed approximately 50,000 Lending Club loan records with a bad-loan rate of about 15.6%.
- Compared a conservative approval policy with an aggressive policy using false negative rate, where a false negative means approving a borrower who later becomes a bad loan.
- The conservative policy reduced bad-loan approval risk from about 42.2% to 0.6%, but with a much lower approval rate, highlighting the trade-off between risk control and business growth.
- Identified three borrower segments:
  - Low-risk borrowers: lower interest rates, DTI, and revolving utilisation, with around 10.4% bad-loan rate.
  - High-leverage borrowers: higher income but heavier debt burden, with around 17.1% bad-loan rate.
  - High-utilisation / financial-stress borrowers: high revolving utilisation and around 18.0% bad-loan rate.
- Built a PyTorch DNN and benchmarked it against logistic regression. Both achieved ROC-AUC around 0.67, showing that the simpler baseline was competitive on this structured tabular dataset.
- Proposed a three-way decision policy: automatically approve low-risk applications, manually review mid-risk cases, and reject high-risk applications.

## Data

The dataset is based on Lending Club loan records. The full raw dataset is not included in this repository because of file size and redistribution considerations.

Expected local files:

- `early_2012_2013_loan_sample_with_outcome.csv`
- `df_clust.csv`

## Skills Demonstrated

- R, Python, PyTorch, pandas, scikit-learn
- A/B testing and hypothesis testing
- Customer / borrower segmentation
- Credit risk analytics
- Model evaluation and threshold tuning
- Business interpretation of statistical and machine learning outputs
