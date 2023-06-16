import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor

def normalize(x):
    x = x.fillna(x.mean())  # Fill NaNs with mean
    if x.std() == 0:
        return x - x.mean()
    else:
        return (x - x.mean()) / x.std()

def process_factors(df):
    df_copy = df.copy()
    df_copy['Size'] = df_copy['MarketCap']
    df_copy['Value'] = df_copy['BookValuePerShare'] / df_copy['LastPrice']
    df_copy['Profitability'] = df_copy['ROE']
    df_copy['Investment'] = df_copy['FreeCashFlow'] / df_copy['MarketCap']
    df_copy['SizeNorm'] = df_copy.groupby('Industry')['Size'].transform(normalize)
    df_copy['ValueNorm'] = df_copy.groupby('Industry')['Value'].transform(normalize)
    df_copy['ProfitabilityNorm'] = df_copy.groupby('Industry')['Profitability'].transform(normalize)
    df_copy['InvestmentNorm'] = df_copy.groupby('Industry')['Investment'].transform(normalize)
    return df_copy

csv_file = 'current_data.csv'
model_file = 'model.pkl'

# Check if the file exists
if not os.path.isfile(csv_file):
    print(f"File {csv_file} not found. Exiting...")
    exit()

# Check if the model exists
if not os.path.isfile(model_file):
    print(f"Model file {model_file} not found. Exiting...")
    exit()

# Load the data
df = pd.read_csv(csv_file)

# Process the data
df = process_factors(df)

# Define features
features = ['SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']

# Check if any features are missing
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print(f"Missing features in data: {missing_features}. Exiting...")
    exit()

# Load the model
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Make predictions
df['Score'] = model.predict(df[features])

# Save the data with the predictions
df.to_csv('scored_data.csv', index=False)

print(f"Predictions saved to 'scored_data.csv'")