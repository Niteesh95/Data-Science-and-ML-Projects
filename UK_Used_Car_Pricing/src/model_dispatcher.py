# model_dispatcher.py
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

models = {
 "linear_regression": LinearRegression(),
 "xgboost_regressor": XGBRegressor(max_depth=7),
}