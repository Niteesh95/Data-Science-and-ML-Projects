# Credit_Card_Fraud_Detection

Libraries used in Sections:
1. EDA - pandas, numpy, matplotlib, seaborn, statsmodels, scaling and sampling, outlier detection and removal, correlation, multicollinearity.
2. Results and Metrics - sklearn, ensemble, metrics.

 In the project I have used models like Decision Trees, XGBoost, Random Forest with various feature engineering techniques to optimise features to keep and omit. The evaluation metrics used are PRF1.

### EDA
The figure below shows the top 10 acount numbers which have the maximum number of frauds against them in the dataset. They account to almost 50% of the fraudulent transaction in the dataset.
<p align="center">
  <img src="https://user-images.githubusercontent.com/60603790/213207891-c547d3cd-e271-405a-9126-9dd8e7e38cb2.png" width=60% height=40% align="center" />
</p>

It can be infered from the bar-graph below that the point-of-sale (POS) entry mode 81 which corresponds to e-commerce transactions account for 70% of the total fraudulent transaction encounteredd in the dataset.
<p align="center">
  <img src="https://user-images.githubusercontent.com/60603790/213208691-a84c665f-2f9f-4021-81cc-b5879eef51d2.png" width=60% height=40% align="center" />
</p>

Merchant country codes 826, 840, & 442 together constitute to 81% of the fraudulent transactions in the dataset.
<p align="center">
  <img src="https://user-images.githubusercontent.com/60603790/213209439-b10339ca-6316-4c0b-bf93-5e6d028f938b.png" width=60% height=40% align="center" />
</p>

### Results and Metrics
Best model with account number and merchant ID as a feature. It was the Decision Tree model.
<p align="center">
  <img src="https://user-images.githubusercontent.com/60603790/213214113-abb13667-693c-4043-b705-f0b0825da457.png" width="500" height="400" />
</p>

Advantages: 
  - The recall must be increased as we are focused on positive outcome (frauds). Increasing recall reduces the frauds being predicted as non-frauds which is ideal case.
  - This model is extremely good at detecting frauds from those sources which has shown fraudulent behaviour at least once.

Best model without account number and merchant ID as a feature. Again, it was the Decision Tree model.
|         | Predicted non-fraud           | Predicted fraud  |
| ------------- |:-------------:| -----:|
| Actual non-fraud      | 14345 | 2459 |
| Actual fraud     | 16      |   63 |
|   | Precision|    Recall |
|0|1.00|0.85|
|1|0.02|0.80|

Advantages: 
  - This model does not rely on account no. and merchant ID.  These models work well even if new account or merchant are created.
