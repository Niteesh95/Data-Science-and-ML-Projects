# Diabetes Prediction App

Data Science project on the Diabetes dataset from Kaggle, dataset can be found [here](https://www.kaggle.com/datasets/mathchi/diabetes-data-set).
Designed and developed a E2E diabetes prediction application using data science concepts and flask. Deployment using Docker containerization and kubernetes on google cloud platform.

## EDA
- Older people in the age range 35-55 have an increased chance of being diagnosed with diabetes.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/60603790/214059776-982aec86-56da-4bfe-b442-374ffb189a54.png" width=60% height=40%/>
</p>

- People having higher glucose content are vulnerable to diabetes diagnosis.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/60603790/214061706-e037712c-16de-4b77-b059-db04ed0f588c.png" width=60% height=40%/>
</p>

## Results and Evaluation
The Decision Tree model was the best model out of all the models used in the project. Below is the confusion matrix of the model. Metrics like PRF1 are used for evaluation.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/60603790/214079811-08f0bac1-0f36-4660-aa16-6eb684f32ff8.png" width=50% height=40%/>
</p>

## Deployment
This model is deployed on a local host using html, flask application. The initial page looks like as shown below.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/60603790/214081115-4096a831-7122-46a5-8c32-7f2da9af3242.png" width=50% height=40%/>
</p>

We enter the required details for the prediction.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/60603790/214081664-97550f12-bfc1-4541-b666-2f326ba5fa99.png" width=50% height=40%/>
</p>

The final page after we click on Predict looks as show below.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/60603790/214082089-638db83d-b58d-4de7-bb51-7d01acb930cd.png" width=50% height=40%/>
</p>

