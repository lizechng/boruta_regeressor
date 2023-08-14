import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
mae = mean_absolute_error(test_labels, pred)
print(f'mae = {mae}')
# mae = 0.01645827832977213


