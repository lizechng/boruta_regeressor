import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.01, 0.1, 1.0]
}

for c in param_grid['C']:
    for ep in param_grid['epsilon']:
        file_name = f'results/svm_train_{c}_{ep}.csv'
        file_data = pd.read_csv(file_name)
        pred = file_data['pred'][:500]
        label = file_data['label'][:500]
        plt.plot(np.arange(len(pred)), pred, 'r--')
        plt.plot(np.arange(len(label)), label, 'g')
        plt.show()

