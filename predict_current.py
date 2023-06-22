import os
import pickle

import pandas as pd


def normalize(x):
    """Normalize a pandas Series"""
    x = x.fillna(x.mean())  # Fill NaNs with mean
    if x.std() == 0:
        return x - x.mean()
    else:
        return (x - x.mean()) / x.std()


def process_factors(df):
    """Process the factors"""
    df_copy = df.copy()
    df_copy['Momentum'] = df_copy['LastPrice'] / \
        df_copy['PreviousYearPrice'] - 1
    df_copy['Size'] = df_copy['MarketCap']
    df_copy['Value'] = df_copy['BookValuePerShare'] / df_copy['LastPrice']
    df_copy['Profitability'] = df_copy['ROE']
    df_copy['Investment'] = df_copy['FreeCashFlow'] / df_copy['MarketCap']
    df_copy['MomentumNorm'] = normalize(df_copy['Momentum'])
    df_copy['SizeNorm'] = normalize(df_copy['Size'])
    df_copy['ValueNorm'] = normalize(df_copy['Value'])
    df_copy['ProfitabilityNorm'] = normalize(df_copy['Profitability'])
    df_copy['InvestmentNorm'] = normalize(df_copy['Investment'])
    return df_copy


csv_file = 'current_data.csv'
model_file = 'gradboost.pkl'

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
features = ['MomentumNorm', 'SizeNorm', 'ValueNorm',
            'ProfitabilityNorm', 'InvestmentNorm']

# Check if any features are missing
missing_features = [
    feature for feature in features if feature not in df.columns]
if missing_features:
    print(f"Missing features in data: {missing_features}. Exiting...")
    exit()

# Load the model
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Make predictions
df['Score'] = model.predict(df[features])
df = df.sort_values('Score', ascending=False)

# Save the data with the predictions
df.to_csv('scored_data.csv', index=False)

print("Predictions saved to 'scored_data.csv'")
