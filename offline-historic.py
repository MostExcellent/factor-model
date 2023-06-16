import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle


def normalize(x):
    x = x.fillna(x.mean())  # Fill NaNs with mean
    if x.std() == 0:
        return x - x.mean()
    else:
        #print(f"x.std(): {x.std()}")
        #print(f"x.mean(): {x.mean()}")
        return (x - x.mean()) / x.std()

# Cap and floor outliers


def cap_and_floor(df, column, lower_percentile, upper_percentile):
    lower, upper = df[column].quantile([lower_percentile, upper_percentile])
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df


def process_factors(df):
    print("Processing factors...")
    df_copy = df.copy()
    # Will add risk free rate when I get that working
    # shift returns back one year
    df_copy['Momentum'] = df_copy['ForwardReturn'].shift(1)
    df_copy['Size'] = df_copy['MarketCap']
    df_copy['Value'] = df_copy['BookValuePerShare'] / df_copy['LastPrice']
    df_copy['Profitability'] = df_copy['ROE']
    df_copy['Investment'] = df_copy['FreeCashFlow'] / df_copy['MarketCap']

    df_copy['MomentumNorm'] = normalize(df_copy['Momentum'])
    df_copy['SizeNorm'] = normalize(df_copy['Size'])
    df_copy['ValueNorm'] = normalize(df_copy['Value'])
    df_copy['ProfitabilityNorm'] = normalize(df_copy['Profitability'])
    df_copy['InvestmentNorm'] = normalize(df_copy['Investment'])

    df_copy['Score'] = df_copy[['MomentumNorm', 'SizeNorm',
                                'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']].sum(axis=1)

    return df_copy


def train_model(x_train, y_train, params=None):
    best_params = params
    if best_params is None:
        # Hyperparameter tuning
        # Define the parameter grid
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }

        rf = RandomForestRegressor()

        # Instantiate the grid search model
        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

        # Fit the grid search to the data
        print("Tuning hyperparameters...")
        grid_search.fit(x_train, y_train)

        # Check the best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

    print("Training model...")
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(x_train, y_train)

    return best_model, best_params


def test_model(rf, x_test, y_test):
    print("Testing model...")
    y_pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

class NaiveModel:
    def fit(self, X, y):
        self.mean = y.mean()
    def predict(self, X):
        return np.full((len(X), ), self.mean)

csv_file = 'data.csv'

# Check if the file exists
if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['LastPrice', 'MarketCap',
                   'BookValuePerShare', 'ROE', 'FreeCashFlow'])
else:
    print(f"File {csv_file} not found. Exiting...")
    exit()

df = cap_and_floor(df, 'LastPrice', 0.01, 0.99)
df = cap_and_floor(df, 'MarketCap', 0.01, 0.99)
df = cap_and_floor(df, 'BookValuePerShare', 0.01, 0.99)
df = cap_and_floor(df, 'ROE', 0.01, 0.99)
df = cap_and_floor(df, 'FreeCashFlow', 0.01, 0.99)

df['ForwardReturn'] = df.groupby('Ticker')['LastPrice'].pct_change(-1)
df.dropna(subset=['ForwardReturn'], inplace=True)
df['ForwardReturnNorm'] = df.groupby(
    'Year')['ForwardReturn'].transform(normalize)

#df_grouped = df.groupby(['Ticker', 'Year']).apply(process_factors)
df_grouped = process_factors(df)
print(df_grouped[['Momentum', 'ForwardReturn', 'Size',
      'Value', 'Profitability', 'Investment']].isnull().sum())
df_grouped.reset_index(inplace=True, drop=True)

features = ['MomentumNorm', 'SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']
#features = ['SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']
target = 'ForwardReturnNorm'

print(df_grouped)
df_grouped = df_grouped.dropna()  # Drop rows with NaN values
print(df_grouped)

x = df_grouped[features]
y = df_grouped[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

rf, best_params = train_model(x_train, y_train)
y_pred, mse, r2 = test_model(rf, x_test, y_test)

with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)
    file.close()

print("MSE: ", mse)
print("R2: ", r2)

naive_model = NaiveModel()
naive_model.fit(x_train, y_train)
y_pred_naive = naive_model.predict(x_test)
mse_naive = mean_squared_error(y_test, y_pred_naive)
r2_naive = r2_score(y_test, y_pred_naive)
print("Naive MSE: ", mse_naive)
print("Naive R2: ", r2_naive)

# Generate plots

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x=rf.feature_importances_, y=features)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins='auto', kde=True)
plt.title('Residuals Distribution')
plt.savefig('residuals.png')

# Distribution of each factor
# for factor in features:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df_grouped[factor], bins='auto', kde=True)
#     plt.title(f'Distribution of {factor}')
#     plt.savefig(f'{factor}_distribution.png')

# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_grouped[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')

# Scatter plot of predicted vs. actual returns
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
# Calculate the coefficients of the line of best fit.
slope, intercept = np.polyfit(y_test, y_pred, 1)
best_fit_line = np.poly1d([slope, intercept])
x_values = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x_values, best_fit_line(x_values), color='red')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title('Predicted vs. Actual Returns')
plt.savefig('predicted_vs_actual.png')

# remove later
#exit()

# Bootstrap analysis
n_samples = 100 #1000

residuals = []
feature_importances = []
mse_vals = []
r2_vals = []

# Naive model
naive_residuals = []
naive_mse_vals = []
naive_r2_vals = []

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Prepare a DataFrame from the training data for bootstrap sampling
train_df = pd.concat([x_train, y_train], axis=1)

# Start the bootstrap analysis
for _ in range(n_samples):
    # Bootstrap sample (with replacement) from the training data only
    sample_df = train_df.sample(frac=1, replace=True)
    x_sample = sample_df[features]
    y_sample = sample_df[target]

    # Train and test model on the bootstrap sample and the untouched test data
    rf, _ = train_model(x_sample, y_sample, best_params)
    y_pred, mse, r2 = test_model(rf, x_test, y_test)

    naive = NaiveModel()
    naive.fit(x_sample, y_sample)
    y_pred_naive = naive.predict(x_test)

    # Record the results
    feature_importances.append(rf.feature_importances_)
    residuals.append(y_test - y_pred)
    mse_vals.append(mse)
    r2_vals.append(r2)
    # Naive model results
    naive_residuals.append(y_test - y_pred_naive)
    naive_mse = mean_squared_error(y_test, y_pred_naive)
    naive_r2 = r2_score(y_test, y_pred_naive)
    naive_mse_vals.append(naive_mse)
    naive_r2_vals.append(naive_r2)

# Now we can calculate the confidence intervals
confidence_level = 0.95

feature_importances_transposed = list(map(list, zip(*feature_importances)))
feature_confidence_intervals = []

for feature_importances in feature_importances_transposed:
    lower = np.percentile(feature_importances,
                          ((1 - confidence_level) / 2) * 100)
    upper = np.percentile(
        feature_importances, (confidence_level + ((1 - confidence_level) / 2)) * 100)
    feature_confidence_intervals.append((lower, upper))

residuals_lower = np.percentile(residuals, ((1 - confidence_level) / 2) * 100)
residuals_upper = np.percentile(
    residuals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
mse_lower = np.percentile(mse_vals, ((1 - confidence_level) / 2) * 100)
mse_upper = np.percentile(
    mse_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
r2_lower = np.percentile(r2_vals, ((1 - confidence_level) / 2) * 100)
r2_upper = np.percentile(
    r2_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)

naive_residuals_lower = np.percentile(naive_residuals, ((1 - confidence_level) / 2) * 100)
naive_residuals_upper = np.percentile(naive_residuals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
naive_mse_lower = np.percentile(naive_mse_vals, ((1 - confidence_level) / 2) * 100)
naive_mse_upper = np.percentile(naive_mse_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
naive_r2_lower = np.percentile(naive_r2_vals, ((1 - confidence_level) / 2) * 100)
naive_r2_upper = np.percentile(naive_r2_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)

print(f"{confidence_level*100}% confidence interval for naive model residuals: ({naive_residuals_lower}, {naive_residuals_upper})")
print(f"{confidence_level*100}% confidence interval for naive model MSE: ({naive_mse_lower}, {naive_mse_upper})")
print(f"{confidence_level*100}% confidence interval for naive model R^2: ({naive_r2_lower}, {naive_r2_upper})")

for i, (lower, upper) in enumerate(feature_confidence_intervals):
    print(f"{confidence_level*100}% confidence interval for feature {i}'s importance: ({lower}, {upper})")

print(f"{confidence_level*100}% confidence interval for residuals: ({residuals_lower}, {residuals_upper})")
print(f"{confidence_level*100}% confidence interval for MSE: ({mse_lower}, {mse_upper})")
print(f"{confidence_level*100}% confidence interval for R^2: ({r2_lower}, {r2_upper})")
