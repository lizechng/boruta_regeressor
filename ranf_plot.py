import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
}

for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        file_name = f'results/ranf_train_{n}_{d}.csv'
        file_data = pd.read_csv(file_name)
        pred = file_data['pred'][:500]
        label = file_data['label'][:500]
        plt.plot(np.arange(len(pred)), pred, 'r--')
        plt.plot(np.arange(len(label)), label, 'g')
        plt.show()
