import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Temperature (°C)': np.random.uniform(60, 100, n_samples),
        'Vibration (mm/s)': np.random.uniform(0.2, 2.5, n_samples),
        'Pressure (Pa)': np.random.uniform(300, 600, n_samples),
        'RPM': np.random.uniform(1000, 2000, n_samples),
        'Maintenance Required': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    return df

def train_and_save_model():
    df = generate_synthetic_data()
    X = df[['Temperature (°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'RPM']]
    y = df['Maintenance Required']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Ensure the models directory exists
    models_dir = os.path.join(os.getcwd(), 'models')  # Using os.getcwd() for current working directory
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'predictive_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()

