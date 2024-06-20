import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

# Load the data
data = pd.read_csv('BBRI.JK.csv')

# Prepare the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]

# Feature engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['Volume'] = np.random.randint(100000, 1000000, size=len(data))  # Replace with actual volume data if available

# Shift the target column to align it with the features
data['Shifted'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Features and labels
X = data[['Close', 'SMA_5', 'SMA_10', 'SMA_20', 'Volume']]
y = data['Shifted']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
