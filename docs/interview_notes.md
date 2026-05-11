# Interview Notes

## 60-Second Project Story

This was a credit risk analytics project using Lending Club loan data. I compared two loan approval policies, segmented borrowers into risk groups, and built a default prediction model. The key business issue was that approving a bad loan is more costly than rejecting a good one, so I focused on false negative rate rather than accuracy. The conservative policy reduced bad-loan approvals from around 42% to below 1%, but it also approved far fewer loans, so the conclusion was not simply to choose the strictest rule. I then used clustering to identify three borrower segments and trained a PyTorch DNN against a logistic regression baseline. The DNN did not outperform the baseline, which taught me that simpler models can be more appropriate for structured tabular credit data.

## Likely Interview Questions

### Why did you use false negative rate?

In this context, a false negative means the policy approves a borrower who later becomes a bad loan. That creates direct credit loss, so it is more important than overall accuracy for the risk-control objective.

### Was Policy A always better?

No. Policy A was much better for reducing default risk, but it had a very low approval rate. A business would need to balance risk reduction against lost revenue from rejecting many potentially good borrowers.

### How did you avoid data leakage?

I excluded post-origination variables such as total payments, recoveries, outstanding principal, and loan status. The analysis focused on features that would be observable at application or origination time.

### Why did the DNN not beat logistic regression?

The dataset is structured tabular data, and many relationships may be captured well by a simpler linear baseline. The result shows why a baseline is important before recommending a more complex model.

### What would you improve next?

I would add stronger cost-sensitive evaluation, tune thresholds using expected profit or loss, test tree-based models such as random forest or XGBoost, and validate the model on a later time period to check stability.

## STAR Answer

**Situation:** A lending company needs to reduce default losses while still approving profitable borrowers.

**Task:** Evaluate approval policies, identify borrower risk segments, and test whether a predictive model can improve decisioning.

**Action:** I cleaned the data, removed leakage variables, compared approval policies using false negative rate, segmented borrowers using clustering, and trained a PyTorch DNN with a logistic regression baseline.

**Result:** The conservative policy reduced bad-loan approval risk from about 42.2% to 0.6%, but with a major approval-rate trade-off. Clustering identified three interpretable borrower segments, and model comparison showed that logistic regression was as competitive as the DNN, supporting a simpler and more explainable baseline.
