import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train():
    train = pd.read_excel('./训练集.xlsx', header=1)

    features = train.iloc[:, :-1]
    labels = train.iloc[:, -1]

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
    }

    # 创建SVM回归模型
    for n in param_grid['n_estimators']:
        for d in param_grid['max_depth']:
            ranf_regressor = RandomForestRegressor(n_estimators=n, max_depth=d)
            ranf_regressor.fit(features, labels)
            pred = ranf_regressor.predict(features)
            res = {'pred': pred,
                   'label': labels}
            res = pd.DataFrame(res)
            res.to_csv(f'results/ranf_train_{n}_{d}.csv')
            mae = mean_absolute_error(labels, pred)
            mse = mean_squared_error(labels, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(labels, pred)
            mape = mean_absolute_percentage_error(labels, pred)
            print(f'{n:03d}\t{d}|{mae:.4f}\t{mse:.4f}\t{rmse:.4f}\t{mape:.4f}\t{r2:.4f}')

def infer():
    selected = ['F1', 'F10']

    # Read Excel files and use the second row as header
    train = pd.read_excel('./训练集.xlsx', header=1)
    test = pd.read_excel('./测试集.xlsx', header=1)

    features = train.loc[:, selected]
    labels = train.iloc[:, -1]

    test_features = test.loc[:, selected]
    test_labels = test.iloc[:, -1]

    rf_regressor = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        random_state=42
    )

    rf_regressor.fit(features, labels)
    pred = rf_regressor.predict(test_features)

    res = {'pred': pred,
           'label': test_labels}
    res = pd.DataFrame(res)
    res.to_csv(f'results/ranf_validate.csv')

    mae = mean_absolute_error(test_labels, pred)
    mse = mean_squared_error(test_labels, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_labels, pred)
    mape = mean_absolute_percentage_error(test_labels, pred)
    print(f'{mae:.4f}\t{mse:.4f}\t{rmse:.4f}\t{mape:.4f}\t{r2:.4f}')

train()
# 050	None|0.0056	0.0001	0.0081	0.6231	0.9276
# 050	10|0.0141	0.0004	0.0189	1.5576	0.6058
# 050	20|0.0078	0.0001	0.0105	0.8589	0.8792
# 100	None|0.0056	0.0001	0.0079	0.6169	0.9306
# 100	10|0.0141	0.0004	0.0189	1.5541	0.6077
# 100	20|0.0078	0.0001	0.0104	0.8546	0.8808
# 150	None|0.0055	0.0001	0.0079	0.6134	0.9317
# 150	10|0.0141	0.0004	0.0189	1.5543	0.6081
# 150	20|0.0078	0.0001	0.0104	0.8526	0.8817

infer()
# 0.0165	0.0005	0.0224	1.8077	0.3718