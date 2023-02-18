# Churn-Analysis
Data Science project on Telcom Churn Prediction dataset. Dataset can be found [here](https://www.kaggle.com/datasets/puja19/telcom-customer-churn).

Libraries: pandas, numpy, matplotlib, seaborn, sklearn, metrics etc.
Tools: python, jupyter.

## EDA
- The customers with partners tend to churn more than the customers who have dependents.
- Customers who do not have any partners churn more when compared to those who have partners.
![image](https://user-images.githubusercontent.com/60603790/214068751-78e160c7-da1d-4a02-b57f-ec7c47c106f9.png)

- The churn rate is more in customers who take month-to-month contracts.
- Customers who take one or two year contracts churn less implying they renew the contract every time.
![image](https://user-images.githubusercontent.com/60603790/214068955-80b2fe50-9cc5-4241-9dc2-2f6d0244e2b6.png)

## Results and Evaluation
The ROC-AUC plot of various models.
![image](https://user-images.githubusercontent.com/60603790/214069899-b0b8967a-f1b2-46fc-b770-895e19d3645f.png)

Logistic Regression was the best model among all the models and experiments carried. The evaluation metrics PRF1 are used.
![image](https://user-images.githubusercontent.com/60603790/214069622-66bbef2c-7bba-4f96-b38d-a7ba563549a7.png)


