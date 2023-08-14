import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold

# Read Excel files and use the second row as header
train = pd.read_excel('./训练集.xlsx', header=1)

features = train.iloc[:, :-1]
labels = train.iloc[:, -1]

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.01, 0.1, 1.0],
    'kernel': ['rbf']
}

# 创建SVM回归模型
svm_regressor = SVR()

# 使用10折交叉验证进行网格搜索
cv = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(svm_regressor, param_grid, scoring='neg_mean_absolute_error', cv=cv)
grid_search.fit(features, labels)

# 打印每种参数组合的MAE结果
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"Parameters: {params}, MAE: {abs(mean_score)}")

# Parameters: {'C': 0.1, 'epsilon': 0.01, 'kernel': 'rbf'}, MAE: 0.01695977434163002
# Parameters: {'C': 0.1, 'epsilon': 0.1, 'kernel': 'rbf'}, MAE: 0.05601301575633079
# Parameters: {'C': 0.1, 'epsilon': 1.0, 'kernel': 'rbf'}, MAE: 0.17671303659411391
# Parameters: {'C': 1.0, 'epsilon': 0.01, 'kernel': 'rbf'}, MAE: 0.016616370242394284
# Parameters: {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'}, MAE: 0.05730778954007103
# Parameters: {'C': 1.0, 'epsilon': 1.0, 'kernel': 'rbf'}, MAE: 0.17671303659411391
# Parameters: {'C': 10.0, 'epsilon': 0.01, 'kernel': 'rbf'}, MAE: 0.016503616684237016
# Parameters: {'C': 10.0, 'epsilon': 0.1, 'kernel': 'rbf'}, MAE: 0.058432692401471545
# Parameters: {'C': 10.0, 'epsilon': 1.0, 'kernel': 'rbf'}, MAE: 0.17671303659411391