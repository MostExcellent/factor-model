import os
import pickle

import pandas as pd
from offline_historic import RFEnsemble # Custom class for ensemble predictions


def normalize(x):
    """Normalize a pandas Series"""
    x = x.fillna(x.mean())  # Fill NaNs with mean
    if x.std() == 0:
        return x - x.mean()
    else:
        return (x - x.mean()) / x.std()


def process_factors(df):
    """
    Process factors for a given pandas dataframe by creating new columns for each factor,
    normalizing each factor, and returning a copy of the dataframe with the new columns.
    """
    print("Processing factors...")
    df_copy = df.copy()

    # shift returns back one year
    df_copy['Momentum'] = (df_copy['PX_LAST'] - df_copy['HIST_TRR_PREV_1YR'])/df_copy['PX_LAST'] - 1
    df_copy['Size'] = df_copy['CUR_MKT_CAP']
    df_copy['Value'] = df_copy['BOOK_VAL_PER_SH'] / df_copy['PX_LAST']
    df_copy['ROE'] = df_copy['RETURN_COM_EQY']
    df_copy['FCF'] = df_copy['CF_FREE_CASH_FLOW'] / df_copy['CUR_MKT_CAP']

    raw_features = ['Momentum', 'Size', 'Value', 'ROE', 'FCF', 'IS_EPS', 'PE_RATIO', 'EPS_GROWTH', 'SALES_GROWTH', 'OPER_MARGIN', 'PROF_MARGIN']
    # Normalize each factor
    for col in raw_features:
        df_copy[f'{col}Norm'] = normalize(df_copy[col])
        
    df_copy = df_copy.dropna()    
    return df_copy

csv_file = 'current_data.csv'
model_file = 'ensemble.pkl'

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
features = features = [col for col in df.columns if col.endswith('Norm')]

# Load the model
print("Loading model...")
model = RFEnsemble().load(model_file)

# Make predictions
scores = model.predict(df[features])
df['Score'] = scores
df = df.sort_values('Score', ascending=False)
# Save the data with the predictions
df.to_csv('scored_data.csv', index=False)

print("Predictions saved to 'scored_data.csv'")
