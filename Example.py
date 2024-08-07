# Example for Data preprocessing (cleaning):
# https://www.kaggle.com/code/pasuvulasaikiran/netflix-appetency-data-cleaning

# Example for Feature selction:

import numpy as np
import pandas as pd
from TabularDataCleaner.FeatureSelection import FeatureSelector

# Example dataset
data = pd.DataFrame({
    'feature1': np.random.choice(['A', 'B', 'C'], 100),
    'feature2': np.random.choice(['X', 'Y', 'Z'], 100),
    'feature3': np.random.rand(100),
    'feature4': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

# Convert categorical features to category dtype
data['feature1'] = data['feature1'].astype('category')
data['feature2'] = data['feature2'].astype('category')

X = data.drop(columns=['target'])
y = data['target']

# Initialize and fit the feature selector
selector = FeatureSelector()
ranking_df = selector.fit(X, y)

# Display feature ranking
print("Feature Ranking:\n", ranking_df)

