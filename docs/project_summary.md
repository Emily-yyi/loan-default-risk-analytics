# Project Summary

## Context

The project studies loan default risk using Lending Club loan data. The analytical goal is to support credit decisioning: approve applicants likely to repay, reject applicants likely to default, and identify cases that need manual review.

## Analytical Workflow

1. Data audit and cleaning
   - Checked missing values, target distribution, feature types, and potential leakage.
   - Removed post-origination fields such as payment, recovery, and outstanding-principal variables.

2. Approval policy comparison
   - Defined two rule-based approval policies:
     - Policy A: conservative approval criteria.
     - Policy B: more aggressive approval criteria.
   - Used false negative rate as the main risk metric because approving a bad loan creates direct financial loss.

3. Borrower segmentation
   - Selected origination-time features such as annual income, DTI, loan amount, interest rate, delinquencies, credit inquiries, open accounts, revolving utilisation, total accounts, and revolving balance.
   - Used hierarchical clustering to guide the number of clusters.
   - Used k-means to profile the full borrower sample into three interpretable risk segments.

4. Default prediction
   - Built a PyTorch DNN model for bad-loan prediction.
   - Created a logistic regression baseline.
   - Applied class weighting to handle the imbalanced target.
   - Tuned thresholds on validation data and evaluated results on a test set.

## Main Findings

- The conservative approval policy greatly reduced false negatives, but at the cost of a very low approval rate.
- Borrower segments revealed different risk drivers: some risk came from leverage, while another segment showed stress through high revolving utilisation.
- The DNN did not materially outperform logistic regression, which is a useful business insight because model complexity should be justified by measurable performance gains.
- A practical deployment approach would combine rules, model scores, and manual review thresholds instead of relying on one strict policy.
