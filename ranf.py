import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Read Excel files and use the second row as header
train = pd.read_excel('./训练集.xlsx', header=1)

features = train.iloc[:, :-1]
labels = train.iloc[:, -1]

# 创建随机森林回归模型
def evaluate_rf_regressor(n_estimators, max_depth):
    rf_regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    mae_scores = -cross_val_score(rf_regressor, features, labels, cv=10, scoring='neg_mean_absolute_error')
    return np.mean(mae_scores)

# 参数候选范围
n_estimators_list = [50, 100, 150]
max_depth_list = [None, 10, 20]

# 参数寻优
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
            mae_mean = evaluate_rf_regressor(n_estimators, max_depth)
            print(f"n_estimators={n_estimators}, max_depth={max_depth}, MAE Mean: {mae_mean}")



# n_estimators=50, max_depth=None, MAE Mean: 0.015570813027317332
# n_estimators=50, max_depth=10, MAE Mean: 0.01565933726226344
# n_estimators=50, max_depth=20, MAE Mean: 0.015466140879405718
# n_estimators=100, max_depth=None, MAE Mean: 0.015488994316539597
# n_estimators=100, max_depth=10, MAE Mean: 0.01564450865820285
# n_estimators=100, max_depth=20, MAE Mean: 0.015407124562592907
# n_estimators=150, max_depth=None, MAE Mean: 0.015454630349293614
# n_estimators=150, max_depth=10, MAE Mean: 0.015637085195504885
# n_estimators=150, max_depth=20, MAE Mean: 0.015383186876103239
