A step-by-step exposition of how to build a bank customer churn prediction model. We have provided the reader with considerable detail for pedagogical purposes.

The main conclusions are:

1. The most important features for churn prediction are `Age`, `NumOfProducts`, `EstimatedSalary`, `CreditScore`, `Balance`, `Geography` and `Gender`.
2. Assuming that the bank's priority is to retain customers (over the incurred costs), our goal is to minimize the number of false negatives and maximize the number of true positives, even if at the expense of false positives. 
Therefore, we change the target metric to $F_2$. The best performing model under (k=5) cross-validation was Support Vector Classifier with f2-score equal to 0.75. The best performing logistic regression had a f2-score of 0.68, compared to the naïve benchmark score of 0.67. 
Both models, however, predict far less false negatives.
3. Restricting the analysis to the relevant features identified by the Kolmogorov-Smirnov and Chi-Squared tests leads to a slight underperformance of the SVC but not of the logistic regression.  
4. The model with the engineered features does not improve performance.
 