# model_dispatcher.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
 "decision_tree_gini": DecisionTreeClassifier(
 criterion='gini', 
 max_depth=3, 
 max_features=6, 
 class_weight={0: 1, 1: 100}
 ),

 "decision_tree_entropy": DecisionTreeClassifier(
 criterion="entropy",
 max_depth=3, 
 max_features=6, 
 class_weight={0: 1, 1: 100}
 ),

 "random_forest_entropy" : RandomForestClassifier(
    criterion='entropy', 
    max_depth=9, 
    class_weight={0: 1, 1: 100}
 ),

 "random_forest_gini" : RandomForestClassifier(
    criterion='gini', 
    max_depth=9, 
    class_weight={0: 1, 1: 100}
 ),

 "xgboost_gbtree" : XGBClassifier(
    booster='gbtree', 
    max_depth=9, 
    learning_rate = 0.1
 ),
}