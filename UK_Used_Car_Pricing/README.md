# Car Pricing - UK
A ML project to analyse and predict the used car prices in the UK. The data set can be found [here](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=hyundi.csv). There are separate csv files for each manufacturer, I have taken 6 different csv files with different manufacturers and merged them to yield a huge dataset of approx. 80k instances and creates a column to determine which model belongs to which manufacturer (make). 

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
 
 # EDA
