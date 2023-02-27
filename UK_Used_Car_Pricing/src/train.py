#!/usr/bin/env python
#  src/train.py

import os
import config
import model_dispatcher

import argparse

import joblib
import pandas as pd
import numpy as np
import datetime



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge, OrthogonalMatchingPursuit, LinearRegression, Lasso, ElasticNet 
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
# from xgboost import XGBRegressor


def run(model, scaling=False):
    df = pd.read_csv(config.TRAINING_FILE)

    #droping the model feature
    df.drop(['model'], axis=1, inplace=True)

    df_dummies = pd.get_dummies(df[['make','transmission', 'fuelType']])

    a = df[['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']]
    df_final = pd.concat([a, df_dummies], axis=1)

    df_final['vehicleAge'] = (datetime.datetime.now().year) - df_final['year']

    df_final.drop(['year'], axis=1, inplace=True)
    df = df_final.copy()
    # scaling features
    if scaling:
        
        scaler = StandardScaler()

        df_final = scaler.fit_transform(df_final)
        df_final=pd.DataFrame(df_final, columns=df.columns)


    X_train, X_test, y_train, y_test = train_test_split(df_final.drop(columns='price'), 
                                                        df_final[['price']],
                                                        test_size=0.2,
                                                    random_state=1)

    # fetch the model from model_dispatcher
    xgb = model_dispatcher.models[model]

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)


    print(f'R^2 Value: %.2f' % xgb.score(X_test,y_test))
    print(f"Mean squared error: %.2f" % mean_squared_error(y_test, xgb.predict(X_test)))
    print(f"Mean absolute error: %.2f" % mean_absolute_error(y_test, xgb.predict(X_test)))

    #save the model
    joblib.dump(xgb, 
                os.path.join(config.MODEL_OUTPUT, f"{model}_scaled_{scaling}.bin"))

if __name__ == "__main__":
     #initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument(
                "--scaling",
                type=bool
             )
    parser.add_argument(
                "--model",
                type=str
                )
                    
    # read the arguments from the command line
    args = parser.parse_args()

# run the fold specified by command line arguments
run(scaling=args.scaling,
    model=args.model)

