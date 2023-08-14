import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

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
mae = mean_absolute_error(test_labels, pred)
print(f'mae = {mae}')
# mae = 0.017389908313946528