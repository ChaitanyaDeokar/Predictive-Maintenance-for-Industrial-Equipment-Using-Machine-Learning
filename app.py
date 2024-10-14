import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, render_template, jsonify

# Add the parent directory of src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import create_app

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('models/predictive_model.pkl')

# Simulate real-time data (in practice, this would come from sensors or a database)
def get_real_time_data():
    now = datetime.now()
    data = {
        'Timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        'Temperature (°C)': np.random.uniform(60, 100),
        'Vibration (mm/s)': np.random.uniform(0.2, 2.5),
        'Pressure (Pa)': np.random.uniform(300, 600),
        'RPM': np.random.uniform(1000, 2000),
    }
    return data

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for the dashboard
@app.route('/dashboard/')
def dashboard():
    return render_template('dashboard.html')

# Route to get real-time data for the dashboard

@app.route('/api/data')
def get_data():
    data = get_real_time_data()
    
    # Generate predictions using the model
    features = pd.DataFrame([[
        data['Temperature (°C)'], 
        data['Vibration (mm/s)'], 
        data['Pressure (Pa)'], 
        data['RPM']
    ]], columns=['Temperature (°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'RPM'])
    
    prediction = model.predict(features)
    data['Maintenance Required'] = 'Yes' if prediction[0] else 'No'
    return jsonify(data)


'''@app.route('/api/data')
def get_data():
    data = get_real_time_data()
    # Generate predictions using the model
    features = np.array([[data['Temperature (°C)'], data['Vibration (mm/s)'], data['Pressure (Pa)'], data['RPM']]])
    prediction = model.predict(features)
    data['Maintenance Required'] = 'Yes' if prediction[0] else 'No'
    return jsonify(data)'''

if __name__ == '__main__':
    app.run(debug=True)
