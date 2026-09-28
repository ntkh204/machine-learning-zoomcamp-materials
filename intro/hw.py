import pandas as pd
import numpy as np

data = pd.read_csv("/workspaces/machine-learning-zoomcamp-materials/car_fuel_efficiency_2026.csv")

# print(data.count())

# print(data['fuel_type'].nunique())

# print(data['fuel_efficiency_mpg'].max())

# med_before = data['horsepower'].median()

# print(med_before)

# mod = data['horsepower'].mode()[0]

# data['horsepower'] = data['horsepower'].fillna(mod)

# med_after = data['horsepower'].median()

# print(med_after)

# Q7

data = data[data['origin'] == 'Asia']

data = data[['vehicle_weight', 'model_year']]

data = data.head(7)

X= data.head(7).to_numpy()

XTX = X.T @ X

XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y
print(w)