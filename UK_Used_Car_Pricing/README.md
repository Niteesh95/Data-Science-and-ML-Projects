# Car Pricing - UK
A ML project to analyse and predict the used car prices in the UK. The data set can be found [here](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=hyundi.csv). There are separate csv files for each manufacturer, I have taken 6 different csv files with different manufacturers and merged them to yield a huge dataset of approx. 80k instances and created a column to determine which model belongs to which manufacturer (make). 

If a separate data for each manufacturer is needed you can download it from the above link or through the *inputs* folder above and filter out the required manufacturers.

## Folder Structure
UK_Used_Car_Pricing </br>
 ┣ .ipynb_checkpoints </br>
 ┣ images </br>
 ┣ inputs </br>
 ┃ ┗ UK_used_cars.csv </br>
 ┣ models </br>
 ┃ ┣ xgb_scaled_None.bin </br>
 ┃ ┗ xgb_scaled_True.bin </br>
 ┣ notebooks </br>
 ┃ ┣ .ipynb_checkpoints </br>
 ┃ ┣ EDA.ipynb </br>
 ┃ ┗ Regression Modelling.ipynb </br>
 ┣ src </br>
 ┃ ┣ __pycache__ </br>
 ┃ ┣ config.py </br>
 ┃ ┣ model_dispatcher.py </br>
 ┃ ┗ train.py </br>
 ┣ .gitattributes </br>
 ┣ .gitignore </br>
 ┗ README.md </br>
 
 ## EDA
 
 **Analysis - Mileage** </br> <br/>
<img src="images/mileage1.jpg" width=400 height=300 align="left"/>
- The lower the mileage, the more the price of the used car.
- If the mileage is <50000, the price can rise as much as >£60k.
<br clear="left"/>
<br/>

**Analysis - Year** <br/><br/>
<img src="images/price by year.jpg" width=400 height=300 align="left">
 - Recently brought cars have a higher price range than the older cars.
 - The price of the cars with model year less than 5 years costs twice the price compared to cars that are purchased more than 5 years ago.
<br clear="left"/>
<br/>

**Analysis - Transmission**<br/><br/>
<img src="images/price hue transmission.jpg" width=400 height=300 align="left">
 - Automatic and semi automatic car prices are more as compared to manual cars. 
 - Most of the manual transmission cars' price range lies between £0-£30k where as the price range for automatic and semi-automatic cars is £0-£45k.
 - More automatic and semi-automatic cars are sold more than £30k
<br clear="left"/>
<br/>

**Analysis - Fuel Type**<br/><br/>
<img src="images/price hue fueltype.jpg" width=400 height=300 align="left">
 - Diesel cars have more price than the petrol cars. 
 - Most diesel car prices range from £15-£30k where as most petrol cars are sold in the price between £0-£15k.
 - Where as hybrid cars, majority of them fall between £15-£30k, not more than that.
<br clear="left"/>
<br/>

**Analysis - Tax**<br/><br/>
<img src="images/tax.jpg" width=400 height=300 align="left">
 - Most of the cars sold in the range £15-£30k have a tax slab of 100-200 units. 
 - Most of the cars priced between £0-£15k have a tax ranging between 0-200.
<br clear="left"/>
<br/>

**Analysis - MPG**<br/><br/>
<img src="images/price by mpg.jpg" width=500 height=350 />
<img src="images/price by mpg 2.jpg" width=500 height=350 />
- Most of the cars sold in the range £15-£30k have a tax slab of 100-200 units. 
- Most of the cars priced between £0-£15k have a tax ranging between 0-200.

**Analysis - Engine Size**<br/><br/>
<img src="images/enginesize by mpg 01.jpg" width=500 height=350 />
<img src="images/enginesize by make.jpg" width=500 height=350 />
<img src="images/price by make.jpg" width=500 height=350 />
<img src="images/mpg by make.jpg" width=500 height=350 />
- Audi, BMW and Mercedes cars with engine size between 1.5L to 2L and with similar MPG as other manufacturers cost at least £10000 higher. 

## Modeling and Results
**Data Cleaning and Feature Engineering**<br/>
- Dropped 'model' feature since it had no relevance.
- Created a 'vehicleAge' feature from 'year' to reduce its multi-collinearity.
- One Hot encoded 'make','transmission', 'fuelType' columns since they were a categorical features.

**Modelling**</br>
- Developed various regression models like linear regression, OLS, ridge, and XGBoost regressor.
- The best model was the XGBoost regressor with R-squared value of 0.94 on the test set. Mean Absolute Error of 1510 on the same test set. This means that the predicted value could be more or less than 1510, so the pricing could be done in that range.

**SHAP**<br/>
<p align='center'>
<img src="images/shap_forceplot.jpg" width=1100 height=100/> </p>
<img src="images/shap_summaryplot.jpg" width=400 height=300 align='left'/>

- SHAP force plot (above) and Summary Plot (left).
- f(x) is the model prediction. Features pushing the prediction higher than the base value are shown in red, those pushing the prediction lower are in blue
- The mean prediction across the full test dataset serves as the basis value. It is the value that would be projected if the current output had no features.
- Features are ordered in descending order by feature importance. Features with high predicting power are shown at the top and the ones with low predicting powers are shown at the bottom.(left plot)
<br clear="left"/>
</br>

**Key takeaways from model from business perspective:**
- The company should focus on features like 'vehicleAge', 'make', 'Transmission type', and 'mpg as these are the ones that push the prediction towards higher side yeilding a higher price.
- 'vehicleAge', 'engineSize', 'Transmission type', 'mileage', 'make', 'mpg', etc are among the top features for the prediction model, re-evaluating these could be a best way to set pricing for the used cars.
