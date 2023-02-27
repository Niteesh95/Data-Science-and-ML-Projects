import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import RocCurveDisplay
from yellowbrick.classifier import ROCAUC

data = pd.read_csv('C:/Users/nitee/OneDrive/Documents/Job Application tasks/Feature Space/data-new/data-new/transactions_obf.csv')
labels_data = pd.read_csv('C:/Users/nitee/OneDrive/Documents/Job Application tasks/Feature Space/data-new/data-new/labels_obf.csv')

lst = [labels_data['eventId'][i] for i in range(len(labels_data))]
a = '(' + '|'.join(lst) + ')'

data['fraudReported'] = data.eventId.str.extract(a, expand=False).fillna(0)

data['fraudReported'] = data['fraudReported'].astype(bool).astype(int)

data['transactionTime'] = pd.to_datetime(data['transactionTime'])

data['yearMonth'] = data['transactionTime'].map(lambda x: str(x.year)+'_'+str(x.month))

data.drop_duplicates(inplace=True)

print(f"% of fraudulent cases: {round(len(data[data['fraudReported'] == 1])/len(data)*100, 2)}")

#===============================================================================
## Feature Engineering
#===============================================================================

train_data = data[data['transactionTime'] < '2017-11-01']
val_data = data.loc[(data['transactionTime'] > '2017-11-01') & (data['transactionTime'] < '2017-12-01')]
test_data = data[data['transactionTime'] >= '2017-12-01']

#generate account number weights
grouped = pd.DataFrame(data.groupby(data['accountNumber'], as_index=False)['fraudReported'].sum().sort_values(by=['fraudReported'], ascending=False))
b = pd.DataFrame(data.groupby(data['accountNumber'], as_index=False)['fraudReported'].count())

accFraud_weight = grouped.merge(b, on='accountNumber').rename(columns={'fraudReported_x':'fraudReported_1', 'fraudReported_y':'fraudReported'})
accFraud_weight['accFraud_weight'] = accFraud_weight['fraudReported_1']/accFraud_weight['fraudReported']

#add account number weights to new columns for corresponding account number
train_data['accFraud_weight'] = train_data['accountNumber'].map(accFraud_weight.set_index('accountNumber')['accFraud_weight'])

val_data['accFraud_weight'] = val_data['accountNumber'].map(accFraud_weight.set_index('accountNumber')['accFraud_weight'])

test_data['accFraud_weight'] = test_data['accountNumber'].map(accFraud_weight.set_index('accountNumber')['accFraud_weight'])

#generate merchant ID weights
grouped = pd.DataFrame(data.groupby(data['merchantId'], as_index=False)['fraudReported'].sum().sort_values(by=['fraudReported'], ascending=False))
b = pd.DataFrame(data.groupby(data['merchantId'], as_index=False)['fraudReported'].count())

merIdFraud_weight = grouped.merge(b, on='merchantId').rename(columns={'fraudReported_x':'fraudReported_1', 'fraudReported_y':'fraudReported'})
merIdFraud_weight['merIdFraud_weight'] = merIdFraud_weight['fraudReported_1']/merIdFraud_weight['fraudReported']
merIdFraud_weight.sort_values(by=['merIdFraud_weight'], ascending=False)

#add mechant ID weights to new columns for corresponding merchant ID
train_data['merIdFraud_weight'] = train_data['merchantId'].map(merIdFraud_weight.set_index('merchantId')['merIdFraud_weight'])

val_data['merIdFraud_weight'] = val_data['merchantId'].map(merIdFraud_weight.set_index('merchantId')['merIdFraud_weight'])

test_data['merIdFraud_weight'] = test_data['merchantId'].map(merIdFraud_weight.set_index('merchantId')['merIdFraud_weight'])

#generate pos entry mode weights
grouped = pd.DataFrame(data.groupby(data['posEntryMode'], as_index=False)['fraudReported'].sum().sort_values(by=['fraudReported'], ascending=False))

b = pd.DataFrame(data.groupby(data['posEntryMode'], as_index=False)['fraudReported'].count())

posFraud_weight = grouped.merge(b, on='posEntryMode').rename(columns={'fraudReported_x':'fraudReported_1', 'fraudReported_y':'fraudReported'})
posFraud_weight['posentrymodeFraud_weight'] = posFraud_weight['fraudReported_1']/posFraud_weight['fraudReported']
posFraud_weight.sort_values(by=['posentrymodeFraud_weight'], ascending=False)

#add pos entry mode weights to new columns for corresponding merchant ID
train_data['posentrymodeFraud_weight'] = train_data['posEntryMode'].map(posFraud_weight.set_index('posEntryMode')['posentrymodeFraud_weight'])

val_data['posentrymodeFraud_weight'] = val_data['posEntryMode'].map(posFraud_weight.set_index('posEntryMode')['posentrymodeFraud_weight'])

test_data['posentrymodeFraud_weight'] = test_data['posEntryMode'].map(posFraud_weight.set_index('posEntryMode')['posentrymodeFraud_weight'])

#generate mcc weights
grouped = pd.DataFrame(data.groupby(data['mcc'], as_index=False)['fraudReported'].sum().sort_values(by=['fraudReported'], ascending=False))
b = pd.DataFrame(data.groupby(data['mcc'], as_index=False)['fraudReported'].count())

mccFraud_weight = grouped.merge(b, on='mcc').rename(columns={'fraudReported_x':'fraudReported_1', 'fraudReported_y':'fraudReported'})
mccFraud_weight['mccFraud_weight'] = mccFraud_weight['fraudReported_1']/mccFraud_weight['fraudReported']
mccFraud_weight.sort_values(by=['mccFraud_weight'], ascending=False)

#add mcc weights to new columns for corresponding merchant ID
train_data['mccFraud_weight'] = train_data['mcc'].map(mccFraud_weight.set_index('mcc')['mccFraud_weight'])

val_data['mccFraud_weight'] = val_data['mcc'].map(mccFraud_weight.set_index('mcc')['mccFraud_weight'])

test_data['mccFraud_weight'] = test_data['mcc'].map(mccFraud_weight.set_index('mcc')['mccFraud_weight'])

#generate merchant country weights
grouped = pd.DataFrame(data.groupby(data['merchantCountry'], as_index=False)['fraudReported'].sum().sort_values(by=['fraudReported'], ascending=False))
b = pd.DataFrame(data.groupby(data['merchantCountry'], as_index=False)['fraudReported'].count())

mercountryFraud_weight = grouped.merge(b, on='merchantCountry').rename(columns={'fraudReported_x':'fraudReported_1', 'fraudReported_y':'fraudReported'})
mercountryFraud_weight['mercountryFraud_weight'] = mercountryFraud_weight['fraudReported_1']/mercountryFraud_weight['fraudReported']
mercountryFraud_weight.sort_values(by=['mercountryFraud_weight'], ascending=False)

#add mechant country weights to new columns for corresponding merchant ID
train_data['mercountryFraud_weight'] = train_data['merchantCountry'].map(mercountryFraud_weight.set_index('merchantCountry')['mercountryFraud_weight'])

val_data['mercountryFraud_weight'] = val_data['merchantCountry'].map(mercountryFraud_weight.set_index('merchantCountry')['mercountryFraud_weight'])

test_data['mercountryFraud_weight'] = test_data['merchantCountry'].map(mercountryFraud_weight.set_index('merchantCountry')['mercountryFraud_weight'])

#===============================================================================
#Without undersampling
#===============================================================================
train_data.drop(['eventId', 'merchantZip', 'yearMonth', 'transactionTime'], axis=1, inplace=True)
val_data.drop(['eventId', 'merchantZip', 'yearMonth', 'transactionTime'], axis=1, inplace=True)
test_data.drop(['eventId', 'merchantZip', 'yearMonth', 'transactionTime'], axis=1, inplace=True)

train_data.drop_duplicates(inplace=True)
val_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)

plt.figure(figsize = (10, 7))
sns.heatmap(data = train_data.corr(), annot = True, linewidth = 1, cmap='Blues')
plt.show()

#===============================================================================
# Train, validation, and test splits
#===============================================================================

X_train = train_data.drop(['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode', 'fraudReported', 'accFraud_weight', 'merIdFraud_weight'], axis=1)
X_val = val_data.drop(['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode','fraudReported', 'accFraud_weight', 'merIdFraud_weight'], axis=1)
X_test = test_data.drop(['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 'posEntryMode','fraudReported', 'accFraud_weight', 'merIdFraud_weight'], axis=1)

y_train = train_data[['fraudReported']]
y_val = val_data[['fraudReported']]
y_test = test_data[['fraudReported']]

X_val.fillna(0.0, axis=1, inplace=True)
X_test.fillna(0.0, axis=1, inplace=True)

#Decision Tree model
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, max_features=6, class_weight={0: 1, 1: 100})
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy : {dtc_train_acc}")
print(f"Test accuracy : {dtc_test_acc}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = dtc.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Confusion Matrix")
plt.grid(visible=False)
plt.show()

print(classification_report(y_test, y_pred))

kfold = StratifiedKFold(n_splits=10)
cv = cross_val_score(dtc,X_val,y_val,cv=kfold)
cv.mean()

#Random Forest model

model = RandomForestClassifier(criterion='entropy', max_depth=9, class_weight={0: 1, 1: 100})
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rfc_train_acc = accuracy_score(y_train, model.predict(X_train))
rfc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy : {rfc_train_acc}")
print(f"Test accuracy : {rfc_test_acc}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.grid(visible=False)
plt.show()

print(classification_report(y_test, y_pred))

#XGBoost model

model = XGBClassifier(learning_rate=0.1, max_depth=9, booster='gbtree')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
xgb_train_acc = accuracy_score(y_train, model.predict(X_train))
xgb_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy : {xgb_train_acc}")
print(f"Test accuracy : {xgb_test_acc}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.grid(visible=False)
plt.show()

print(classification_report(y_test, y_pred))

#===============================================================================
# # Undersampling
#===============================================================================
df_fraud_rows = train_data[train_data['fraudReported'] == 1]

accno = pd.DataFrame(train_data.groupby(train_data['accountNumber'])['fraudReported'].sum()).reset_index()
acc_with0Fraud = accno[accno['fraudReported'] == 0]

merid = pd.DataFrame(train_data.groupby(train_data['merchantId'])['fraudReported'].sum()).reset_index()
merid_with0Fraud = merid[merid['fraudReported'] == 0]

pos = pd.DataFrame(train_data.groupby(train_data['posEntryMode'])['fraudReported'].sum()).reset_index()
pos_with0Fraud = pos[pos['fraudReported'] == 0]

mcc = pd.DataFrame(train_data.groupby(train_data['mcc'])['fraudReported'].sum()).reset_index()
mcc_with0Fraud = mcc[mcc['fraudReported'] == 0]

df_fraud = train_data[~((train_data['merchantId'].isin(merid_with0Fraud['merchantId'].tolist())) & (train_data['accountNumber'].isin(acc_with0Fraud['accountNumber'].tolist())))]

only_non_fraud = df_fraud[df_fraud['fraudReported'] == 0].sample(frac=0.15)

only_fraud = df_fraud[df_fraud['fraudReported'] == 1]
undersampled_df = pd.concat([only_non_fraud, only_fraud])

X_train = undersampled_df.drop(['accountNumber','merchantId', 'mcc', 'posEntryMode', 'merchantCountry', 'fraudReported', 'accFraud_weight','merIdFraud_weight'], axis=1)
X_val = val_data.drop(['accountNumber','merchantId', 'mcc', 'posEntryMode', 'merchantCountry', 'fraudReported', 'accFraud_weight','merIdFraud_weight'], axis=1)
X_test = test_data.drop(['accountNumber','merchantId', 'mcc', 'posEntryMode', 'merchantCountry', 'fraudReported', 'accFraud_weight','merIdFraud_weight'], axis=1)

y_train = undersampled_df[['fraudReported']]
y_val = val_data[['fraudReported']]
y_test = test_data[['fraudReported']]

X_val.fillna(0.0, axis=1, inplace=True)
X_test.fillna(0.0, axis=1, inplace=True)

#Decision Tree model - best model in the previous experiment hence choosen
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, max_features=6)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy : {dtc_train_acc}")
print(f"Test accuracy : {dtc_test_acc}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = dtc.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Decision Tree Confusion Matrix")
plt.grid(visible=False)
plt.show()

print(classification_report(y_test, y_pred))
