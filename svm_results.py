import pandas as pd
from sklearn.svm import SVR
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
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1, 1.0],
        'kernel': ['rbf']
    }

    # 创建SVM回归模型
    for c in param_grid['C']:
        for ep in param_grid['epsilon']:
            svm_regressor = SVR(C=c, epsilon=ep)
            svm_regressor.fit(features, labels)
            pred = svm_regressor.predict(features)
            res = {'pred': pred,
                   'label': labels}
            res = pd.DataFrame(res)
            res.to_csv(f'results/svm_train_{c}_{ep}.csv')
            mae = mean_absolute_error(labels, pred)
            mse = mean_squared_error(labels, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(labels, pred)
            mape = mean_absolute_percentage_error(labels, pred)
            print(f'{c:02.2f}\t{ep:02.2f}|{mae:.4f}\t{mse:.4f}\t{rmse:.4f}\t{mape:.4f}\t{r2:.4f}')

def infer():
    selected = ['F1', 'F10']

    # Read Excel files and use the second row as header
    train = pd.read_excel('./训练集.xlsx', header=1)
    test = pd.read_excel('./测试集.xlsx', header=1)

    features = train.loc[:, selected]
    labels = train.iloc[:, -1]

    test_features = test.loc[:, selected]
    test_labels = test.iloc[:, -1]

    svm_regressor = SVR(C=10.0, epsilon=0.01, kernel='rbf')

    svm_regressor.fit(features, labels)

    pred = svm_regressor.predict(test_features)

    res = {'pred': pred,
           'label': test_labels}
    res = pd.DataFrame(res)
    res.to_csv(f'results/svm_validate.csv')

    mae = mean_absolute_error(test_labels, pred)
    mse = mean_squared_error(test_labels, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_labels, pred)
    mape = mean_absolute_percentage_error(test_labels, pred)
    print(f'{mae:.4f}\t{mse:.4f}\t{rmse:.4f}\t{mape:.4f}\t{r2:.4f}')

# train()
# 0.10	0.01|0.0169	0.0006	0.0241	1.8857	0.3581
# 0.10	0.10|0.0561	0.0036	0.0604	6.0410	-3.0135
# 0.10	1.00|0.1810	0.0336	0.1834	19.5667	-36.0485
# 1.00	0.01|0.0166	0.0006	0.0236	1.8457	0.3887
# 1.00	0.10|0.0573	0.0038	0.0614	6.1644	-3.1473
# 1.00	1.00|0.1810	0.0336	0.1834	19.5667	-36.0485
# 10.00	0.01|0.0164	0.0005	0.0233	1.8286	0.4042
# 10.00	0.10|0.0593	0.0040	0.0632	6.3816	-3.4040
# 10.00	1.00|0.1810	0.0336	0.1834	19.5667	-36.0485


infer()
# 0.0174	0.0006	0.0239	1.9209	0.2864