# Cardiovascular Disease Diagnosis Prediction
This repo is a open-source project for the Kaggle cardiovascular dataset available [here](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

Sections:
1. Importing necessary libraries.
2. EDA, data cleaning, data impuation, correlation, multicollinearity, outlier detection and handling.
3. Data Analysis, visualization, and data preparation.
4. Data modelling with various ML algorithms and prediction.

## EDA
Below graph shows the density plot of age with respect to CVD diagnosis. We can infer that indivuduals aged 55-65 have a higher risk of being diagnosed with CVD.
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/214552273-935abd04-29e0-42f4-ac62-6892e12f986f.png' width='500' height='300' />
</p>

The correlation plot of independent features is shown below.
- The bmi, which is a feature created from weight and height variables, has a strong correlation with weight variable.
- The ap_hi and ap_lo are strongly correlated with each other with 73% correlation.
<p align='center'>
  <img src='https://user-images.githubusercontent.com/60603790/214559439-6578a7e7-e7d5-48a8-976a-3d0dc257b0b8.png' width='500' height='350' />
</p>

## Results and Evaluation
The random forest model outperforms the other models in all the aspects, accuracy, f1-score, and cross-validation accuracy. The AUC of random forest is also more compared to the other models. Therefore, we can say that the random forest model is the best model for this dataset.
  <p align='center'/>
    <img src='https://user-images.githubusercontent.com/60603790/214560280-8cb7711e-dc08-4caf-841b-ae64415e237a.png' width='400' height='200' />
    <img src='https://user-images.githubusercontent.com/60603790/214560679-6f23451a-c3e6-4459-bf77-888433787f2a.png' width='400' height='300' />
  </p>
  
 ## Deployment
 This model is deployed on a local host using html, flask application. The initial page looks like as shown below.
 <p align='center'/>
    <img src='https://user-images.githubusercontent.com/60603790/214571945-a4b50aa5-5942-4b45-9e41-dd21d35a759d.png' width='300' height='500' />
  </p>
  
We enter the required details for the prediction.
<p align='center'/>
    <img src='https://user-images.githubusercontent.com/60603790/214572134-198333eb-855f-4e0e-83b8-431901e51b00.png' width='300' height='500' />
  </p>

The final page after we click on Predict looks as show below.
<p align='center'/>
    <img src='https://user-images.githubusercontent.com/60603790/214572250-e7feac72-122d-4667-9465-c727c4c0452e.png' width='300' height='500' />
  </p>





