#importing packages
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lightgbm

config = {
    "input_dir":'/kaggle/working/',
    "n_folds":5,
    "seeds":40,
    "boosting_type":'dart',
    "target":'target',
    "metric":'binary_logloss',
    "cat_features":[
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    
}

#seeding
random.seed(config["seeds"])
np.random.seed(config["seeds"])

# Metrics 
def metric(y_true, y_pred):
    # Creating labels
    labels = np.array([y_true, y_pred]).T
    labels = labels[np.argsort(-y_pred)]
    weights = np.where(labels[:, 0] == 0, 20, 1)

    # Get 4% of the labelst
    cutVals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]

    # Calculating the accuracy
    topFourAccuracy = np.sum(cutVals[:, 0]) / np.sum(labels[:, 0])

    
    gi = [0, 0]
    for i in [1, 0]:
        # Creating labels array
        labels = np.array([y_true, y_pred]).T
        labels = labels[np.argsort(-y_pred if i else -y_true)]

        weight = np.where(labels[:, 0] == 0, 20, 1)

        random_weights = np.cumsum(weight / np.sum(weight))
        
        totalPos = np.sum(labels[:, 0] * weight)
        cumPosFound = np.cumsum(labels[:, 0] * weight)
        lorentz = cumPosFound / totalPos

        # Calculate gini
        gi[i] = np.sum((lorentz - random_weights) * weight)

    # Returning final metrics
    return 0.5 * (gi[1] / gi[0] + topFourAccuracy)

def lightgbm_metric(y_pred, y_true):
    y_true = y_true.get_label()
    metric_value = metric(y_true, y_pred)
    return 'amex_metric', metric_value, True


# data preprocessing
def get_difference(data, num_feat):
    #Function which calculate the difference of numeric features in a dataframe and grouped by 'customer_ID'
    df = []
    customersID = []
    
    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        diff_df = df[num_feat].diff(1).iloc[[-1]].values.astype(np.float32)
        df.append(diff_df)
        customersID.append(customer_id)
    df = np.concatenate(df, axis=0)
    df = pd.DataFrame(df, columns=[col + "_diff1" for col in df[num_feat].columns])
    df["customer_ID"] = customersID
    return df


def reading_preprocess_data():
    #Function to read, preprocess, and aggregate data

    print("++++++++++++Reading data+++++++++++++++++++")
    train = pd.read_csv("/kaggle/input/amex-default-prediction/train_data.csv")
    features = train.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
    cateFeatures = config["cat_features"]
    numFeatures = [column for column in features if column not in cateFeatures]


    print("++++++++++++++++++++training started++++++++++++")
    trainNumAgg = train.groupby("customer_ID")[numFeatures].agg(["mean", "std", "min", "max", "last"])
    trainNumAgg.columns = ["_".join(x) for x in trainNumAgg.columns]
    trainNumAgg.reset_index(inplace=True)


    # Aggregating categorical features by customer_ID
    trainCatAgg = train.groupby("customer_ID")[cateFeatures].agg(["count", "last", "nunique"])
    trainCatAgg.columns = ["_".join(x) for x in trainCatAgg.columns]
    trainCatAgg.reset_index(inplace=True)

    # reading training labels
    trainLabels = pd.read_csv("/kaggle/input/amex-default-prediction/train_labels.csv")

    cols = list(trainNumAgg.dtypes[trainNumAgg.dtypes == "float64"].index)
    for col in tqdm(cols):
        trainNumAgg[col] = trainNumAgg[col].astype(np.float32)
    cols = list(trainCatAgg.dtypes[trainCatAgg.dtypes == "int64"].index)
    for col in tqdm(cols):
        trainCatAgg[col] = trainCatAgg[col].astype(np.int32)

    # Calculating differences of numeric features by customer_ID
    trainDiff = get_difference(train, numFeatures)

    # Merging the aggregated
    train = trainNumAgg.merge(
        trainCatAgg, how="inner", on="customer_ID"
    ).merge(trainDiff, how="inner", on="customer_ID").merge(
        trainLabels, how="inner", on="customer_ID"
    )


    # Reading test data
    test = pd.read_parquet("/kaggle/input/amex-default-prediction/test_data.csv")
    print("+++++++++++++++++++++Started test feature engineering++++++++++++++++++++++")
    testNumAgg = test.groupby("customer_ID")[numFeatures].agg(["mean", "std", "min", "max", "last"])
    testNumAgg.columns = ["_".join(x) for x in testNumAgg.columns]
    testNumAgg.reset_index(inplace=True)
    testCatAgg = test.groupby("customer_ID")[cateFeatures].agg(["count", "last", "nunique"])
    testCatAgg.columns = ["_".join(x) for x in testCatAgg.columns]
    testCatAgg.reset_index(inplace=True)
    testDiff = get_difference(test, numFeatures)
    test = testNumAgg.merge(testCatAgg, how="inner", on="customer_ID").merge(
        testDiff, how="inner", on="customer_ID"
    )
    return train, test

train,test = reading_preprocess_data()


def train_evaluate(train, test):
# Labeling encode categorical features
    catCols = config["cat_features"]
    catCols = [f"{col}_last" for col in catCols]
    for coloumn in catCols:
        train[coloumn] = train[coloumn].astype('category')
        test[coloumn] = test[coloumn].astype('category')

    floatCols = train.select_dtypes(include=['float']).columns
    floatCols = [coloumn for coloumn in floatCols if 'last' in coloumn]
    train[floatCols] = train[floatCols].round(2)
    test[floatCols] = test[floatCols].round(2)

    # difference between mean and last
    numCols = [coloumn for coloumn in train.columns if 'last' in coloumn]
    numCols = [coloumn[:-5] for coloumn in numCols if 'round' not in coloumn]
    for coloumn in numCols:
        train[f'{coloumn}_last_mean_diff'] = train[f'{coloumn}_last'] - train[f'{coloumn}_mean']
        test[f'{coloumn}_last_mean_diff'] = test[f'{coloumn}_last'] - test[f'{coloumn}_mean']

    floatCols = train.select_dtypes(include=['float']).columns
    train[floatCols] = train[floatCols].astype(np.float16)
    test[floatCols] = test[floatCols].astype(np.float16)

    # feature list
    features = [coloumn for coloumn in train.columns if coloumn not in ['customer_ID', config.target]]

    # Model parameters
    params = {
        'objective': 'binary',
        'metric': config.metric,
        'boosting': config.boosting_type,
        'seed': config.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }

    testPredictions = np.zeros(len(test))
    oofPredictions = np.zeros(len(train))

    
    kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[config.target])):
        print(f'\nTraining fold {fold} with {len(features)} features++++++++++')
        xTrain, xVal = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        yTrain, yVal = train[config.target].iloc[trn_ind], train[config.target].iloc[val_ind]
        lightgbm_train = lightgbm.Dataset(xTrain, yTrain, categorical_feature=catCols)
        lightgbm_val = lightgbm.Dataset(xVal, yVal, categorical_feature=catCols)
        model = lightgbm.train(params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_val],
                        valid_names=['train', 'val'], num_boost_round=1000,
                        early_stopping_rounds=50, verbose_eval=50,
                        feval=lightgbm_metric)

        oofPredictions[val_ind] = model.predict(xVal)
        testPredictions += model.predict(test[features]) / config.n_folds
        score = metric(yVal,model.predict(xVal))

    score = metric(train[config.target], oofPredictions)
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': testPredictions})
    test_df.to_csv(f'/kaggle/working/submission.csv', index = False)
    
train_evaluate(train, test)