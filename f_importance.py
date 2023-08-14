import matplotlib.pyplot as plt
from borutashap import BorutaShap
import numpy as np
import pandas as pd

train = pd.read_excel('./训练集.xlsx', header=1)

features = train.iloc[:500, :-1]
labels = train.iloc[:500, -1]

feature_selector = BorutaShap(importance_measure='shap', classification=False)

feature_selector.fit(X=features, y=labels, n_trials=60, sample=False,
                     train_or_test='train', normalize=False, verbose=True)

feature_selector.plot(which_features='all')

history = feature_selector.history_x.iloc[1:, :-4]
for i in range(history.shape[1]):
    if history.columns[i] in feature_selector.accepted:
        plt.plot(np.arange(1, 61), history.iloc[:, i], 'g')
    else:
        plt.plot(np.arange(1, 61), history.iloc[:, i], 'r--')
plt.savefig('history.png')
plt.show()


print(feature_selector.accepted)