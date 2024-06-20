from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data
data = pd.read_csv('BBRI.JK.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature engineering function
def add_features(data):
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Volume'] = np.random.randint(100000, 1000000, size=len(data))  # Replace with actual volume data if available
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    json_data = request.get_json()
    date_str = json_data.get('date')

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    # Get the latest available date in the dataset
    latest_date = data.index.max()
    data_with_features = add_features(data.copy())
    
    if date <= latest_date:
        if date in data_with_features.index:
            last_close = data_with_features.loc[date][['Close', 'SMA_5', 'SMA_10', 'SMA_20', 'Volume']].values
        else:
            return jsonify({'error': 'Date not found in the data'}), 404
    else:
        last_close = data_with_features.iloc[-1][['Close', 'SMA_5', 'SMA_10', 'SMA_20', 'Volume']].values
        while latest_date < date:
            forecasted_price = model.predict([last_close])[0]
            latest_date += pd.Timedelta(days=1)
            new_row = {
                'Close': forecasted_price,
                'SMA_5': (last_close[0] * 4 + forecasted_price) / 5,
                'SMA_10': (last_close[0] * 9 + forecasted_price) / 10,
                'SMA_20': (last_close[0] * 19 + forecasted_price) / 20,
                'Volume': np.random.randint(100000, 1000000)
            }
            last_close = np.array(list(new_row.values()))

    forecasted_price = model.predict([last_close])[0]
    return jsonify({'forecasted_price': forecasted_price})

if __name__ == '__main__':
    app.run(debug=True)
