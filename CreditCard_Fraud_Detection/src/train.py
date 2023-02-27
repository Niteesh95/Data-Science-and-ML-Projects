import os
import argparse

import config
import model_dispatcher

import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, classification_report

def run(model):
    data = pd.read_csv(config.TRAIN_DATA)
    labels_data = pd.read_csv(config.LABELS_DATA)

    lst = [labels_data['eventId'][i] for i in range(len(labels_data))]
    a = '(' + '|'.join(lst) + ')'

    data['fraudReported'] = data.eventId.str.extract(a, expand=False).fillna(0)

    data['fraudReported'] = data['fraudReported'].astype(bool).astype(int)

    data['transactionTime'] = pd.to_datetime(data['transactionTime'])

    data['yearMonth'] = data['transactionTime'].map(lambda x: str(x.year)+'_'+str(x.month))

    data.drop_duplicates(inplace=True)

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

    #modelling
    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    print(f"{model} Training accuracy : {train_acc}")
    print(f"{model} Test accuracy : {test_acc}")
    print(classification_report(y_test, y_pred))

    #save the model
    joblib.dump(clf, 
                os.path.join(config.MODEL_OUTPUT, f"CCFraud_{model}.bin"))

if __name__ == "__main__":
     #initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    parser.add_argument(
                "--model",
                type=str
                )
                    
    # read the arguments from the command line
    args = parser.parse_args()

# run the fold specified by command line arguments
run(model=args.model)

