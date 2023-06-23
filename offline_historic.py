
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


def normalize(x):
    """
    Normalize a pandas series by subtracting the mean and dividing by the standard deviation.
    If the standard deviation is 0, return the series with the mean subtracted.
    """
    x = x.fillna(x.mean())  # Fill NaNs with mean
    if x.std() == 0:
        return x - x.mean()
    else:
        return (x - x.mean()) / x.std()


def cap_and_floor(df, column, lower_percentile, upper_percentile):
    """
    Cap and floor outliers in a pandas dataframe column based on percentiles.
    """
    lower, upper = df[column].quantile([lower_percentile, upper_percentile])
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df


def process_factors(df):
    """
    Process factors for a given pandas dataframe by creating new columns for each factor,
    normalizing each factor within each year, and returning a copy of the dataframe with the new columns.
    """
    print("Processing factors...")
    df_copy = df.copy()

    # shift returns back one year
    df_copy['Momentum'] = df_copy.groupby(
        'Ticker')['ForwardReturn'].transform(lambda x: x.shift(1))
    df_copy['Size'] = df_copy['MarketCap']
    df_copy['Value'] = df_copy['BookValuePerShare'] / df_copy['LastPrice']
    df_copy['Profitability'] = df_copy['ROE']
    df_copy['Investment'] = df_copy['FreeCashFlow'] / df_copy['MarketCap']

    # Normalize within each industry
    df_copy['MomentumNorm'] = df_copy.groupby(
        ['Year'])['Momentum'].transform(normalize)
    df_copy['SizeNorm'] = df_copy.groupby(
        ['Year'])['Size'].transform(normalize)
    df_copy['ValueNorm'] = df_copy.groupby(
        ['Year'])['Value'].transform(normalize)
    df_copy['ProfitabilityNorm'] = df_copy.groupby(
        ['Year'])['Profitability'].transform(normalize)
    df_copy['InvestmentNorm'] = df_copy.groupby(
        ['Year'])['Investment'].transform(normalize)

    return df_copy


class RFEnsemble:
    """
    Random forest ensemble class with hyperparameter tuning, training, and prediction methods.
    """

    def __init__(self, regressor, num_models=5, params=None):
        self.num_models = num_models
        self.params = params
        self.regressor = regressor
        self.models = []
        self.feature_importances = []

    def optimize_params(self, x, y, method=GridSearchCV, param_grid=None):
        """
        Optimize the hyperparameters of the random forest regressor using the given method and parameter grid.
        """
        if param_grid == None:
            param_grid = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt']
            }
        optimizer = method(estimator=self.regressor,
                           param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        print("Tuning hyperparameters...")
        optimizer.fit(x, y)
        best_params = optimizer.best_params_
        print(f"Best parameters: {best_params}")
        self.params = best_params
        return best_params

    def train(self, x_train, y_train, get_params=False):
        """
        Train the random forest ensemble on the given training data.
        """
        if self.params is None:
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

            self.params = self.optimize_params(
                x_train, y_train, rf, GridSearchCV, param_grid)

        print("Training models...")
        for _ in range(self.num_models):
            model = RandomForestRegressor(**self.params)
            model.fit(x_train, y_train)
            self.models.append(model)
            self.feature_importances.append(model.feature_importances_)
        if get_params:
            return self.params

    def get_mean_feature_importances(self):
        """
        Get the mean feature importances of the trained random forest ensemble.
        """
        return np.mean(self.feature_importances, axis=0)

    def predict(self, x_test):
        """
        Predict the output for the given input using the trained random forest ensemble.
        """
        print("Testing model...")
        predictions = []
        for model in self.models:
            y_pred = model.predict(x_test)
            predictions.append(y_pred)

        # Average predictions
        y_pred_avg = np.mean(predictions, axis=0)
        return y_pred_avg


class LinearModel:
    """
    Linear regression model class with fit, predict, and test methods.
    """

    def fit(self, X, y):
        """
        Fit the linear regression model to the given data.
        """
        y = y.flatten()
        self.coef = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predict the output for the given input using the trained linear regression model.
        """
        return X @ self.coef

    def test(self, X, y):
        """
        Test the trained linear regression model on the given test data and return the mean squared error and R-squared.
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2


class NaiveModel:
    """
    Naive model class to simulate random chance with fit, predict, and test methods.
    """

    def fit(self, y):
        """
        Fit the naive model to the given data.
        """
        self.mean = y.mean()

    def predict(self, x):
        """
        Predict the output for the given input using the naive model.
        """
        return np.full((len(x), ), self.mean)

    def test(self, x, y):
        """
        Test the naive model on the given test data and return the mean squared error and R-squared.
        """
        y_pred = self.predict(x)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2


csv_file = 'data.csv'


# Check if the file exists
if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['LastPrice', 'MarketCap',
                   'BookValuePerShare', 'ROE', 'FreeCashFlow'])
else:
    print(f"File {csv_file} not found. Exiting...")
    exit()

def clean_data(df):
    """
    Apply cap_and_floor function to specified columns in the given dataframe.
    """
    df = cap_and_floor(df, 'LastPrice', 0.01, 0.99)
    df = cap_and_floor(df, 'MarketCap', 0.01, 0.99)
    df = cap_and_floor(df, 'BookValuePerShare', 0.01, 0.99)
    df = cap_and_floor(df, 'ROE', 0.01, 0.99)
    df = cap_and_floor(df, 'FreeCashFlow', 0.01, 0.99)
    return df


# Check if the file exists
if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['LastPrice', 'MarketCap',
                   'BookValuePerShare', 'ROE', 'FreeCashFlow'])
    #df = clean_data(df)
else:
    print(f"File {csv_file} not found. Exiting...")
    exit()

df = df.sort_values(by=['Ticker', 'Year'])
df['ForwardReturn'] = df.groupby('Ticker')['LastPrice'].pct_change(-1)
df['ForwardReturnNorm'] = df.groupby(
    'Year')['ForwardReturn'].transform(normalize)

# Log returns
df = df.sort_values(by=['Ticker', 'Year'])
df['LogReturn'] = np.log(df['ForwardReturn'] + 1)
df.dropna(subset=['LogReturn', 'ForwardReturn'], inplace=True)
df['LogReturnNorm'] = df.groupby('Year')['LogReturn'].transform(normalize)

#df_grouped = df.groupby(['Ticker', 'Year']).apply(process_factors)
df_grouped = process_factors(df)
print(df_grouped[['Momentum', 'ForwardReturn', 'Size',
      'Value', 'Profitability', 'Investment']].isnull().sum())
df_grouped.reset_index(inplace=True, drop=True)

features = ['MomentumNorm', 'SizeNorm', 'ValueNorm',
            'ProfitabilityNorm', 'InvestmentNorm']
#features = ['Momentum', 'Size', 'Value', 'Profitability', 'Investment']
target = 'LogReturnNorm'
#target = 'LogReturn'

print(df_grouped)
df_grouped = df_grouped.dropna()  # Drop rows with NaN values
print(df_grouped)

x = df_grouped[features]
y = df_grouped[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model = RFEnsemble()
model = model.train(x_train, y_train)
y_pred, mse, r2 = RFEnsemble.test(model, x_test, y_test)

with open('ensemble.pkl', 'wb') as file:
    pickle.dump(model, file)
    file.close()

model_params = model.params

print("MSE: ", mse)
print("R2: ", r2)

linear_model = LinearModel()
linear_model.fit(x_train.values, y_train.values)
y_pred_linear = linear_model.predict(x_test)
mse_linear, r2_linear = linear_model.test_model(x_test, y_test)
print("Linear MSE: ", mse_linear)
print("Linear R2: ", r2_linear)

naive_model = NaiveModel()
naive_model.fit(y_train)
mse_naive, r2_naive = naive_model.test_model(x_test, y_test)
print("Naive MSE: ", mse_naive)
print("Naive R2: ", r2_naive)

with open('initial_test_results.txt', 'w') as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"R2: {r2}\n")
    file.write(f"Linear MSE: {mse_linear}\n")
    file.write(f"Linear R2: {r2_linear}\n")
    file.write(f"Naive MSE: {mse_naive}\n")
    file.write(f"Naive R2: {r2_naive}\n")
    file.close()

# Generate plots

# Feature importance plot
feature_importances = model.get_feature_importances()
feature_importances_avg = np.mean(feature_importances, axis=0)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_avg, y=features)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins='auto', kde=True)
plt.title('Residuals Distribution')
plt.savefig('residuals.png')

# Plot residuals as a function of predicted returns
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.title('Residuals vs. Predicted Returns')
plt.savefig('residuals_vs_predicted_returns.png')

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
# exit()

# Bootstrap analysis
n_samples = 1000

residuals = []
feature_importances = []
mse_vals = []
r2_vals = []

# Linear model
linear_residuals = []
linear_mse_vals = []
linear_r2_vals = []

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
    ensemble = RFEnsemble(num_models=5)
    ensemble.fit(x_sample, y_sample)
    y_pred = ensemble.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    linear = LinearModel()
    linear.fit(x_sample.values, y_sample.values)
    y_pred_linear = linear.predict(x_test)

    naive = NaiveModel()
    naive.fit(y_sample)
    y_pred_naive = naive.predict(x_test)

    # Record the results
    feature_importances.append(ensemble.feature_importances_)
    residuals.append(y_test - y_pred)
    mse_vals.append(mse)
    r2_vals.append(r2)

    # Linear model results
    linear_residuals.append(y_test - y_pred_linear)
    linear_mse = mean_squared_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)
    linear_mse_vals.append(linear_mse)
    linear_r2_vals.append(linear_r2)

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

linear_residuals_lower = np.percentile(
    linear_residuals, ((1 - confidence_level) / 2) * 100)
linear_residuals_upper = np.percentile(
    linear_residuals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
linear_mse_lower = np.percentile(
    linear_mse_vals, ((1 - confidence_level) / 2) * 100)
linear_mse_upper = np.percentile(
    linear_mse_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
linear_r2_lower = np.percentile(
    linear_r2_vals, ((1 - confidence_level) / 2) * 100)
linear_r2_upper = np.percentile(
    linear_r2_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)

naive_residuals_lower = np.percentile(
    naive_residuals, ((1 - confidence_level) / 2) * 100)
naive_residuals_upper = np.percentile(
    naive_residuals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
naive_mse_lower = np.percentile(
    naive_mse_vals, ((1 - confidence_level) / 2) * 100)
naive_mse_upper = np.percentile(
    naive_mse_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
naive_r2_lower = np.percentile(
    naive_r2_vals, ((1 - confidence_level) / 2) * 100)
naive_r2_upper = np.percentile(
    naive_r2_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)

print(f"{confidence_level*100}% confidence interval for naive model residuals: ({naive_residuals_lower}, {naive_residuals_upper})")
print(f"{confidence_level*100}% confidence interval for naive model MSE: ({naive_mse_lower}, {naive_mse_upper})")
print(f"{confidence_level*100}% confidence interval for naive model R^2: ({naive_r2_lower}, {naive_r2_upper})")

print(f"{confidence_level*100}% confidence interval for linear model residuals: ({linear_residuals_lower}, {linear_residuals_upper})")
print(f"{confidence_level*100}% confidence interval for linear model MSE: ({linear_mse_lower}, {linear_mse_upper})")
print(f"{confidence_level*100}% confidence interval for linear model R^2: ({linear_r2_lower}, {linear_r2_upper})")

for i, (lower, upper) in enumerate(feature_confidence_intervals):
    print(f"{confidence_level*100}% confidence interval for feature {i}'s importance: ({lower}, {upper})")

print(f"{confidence_level*100}% confidence interval for residuals: ({residuals_lower}, {residuals_upper})")
print(f"{confidence_level*100}% confidence interval for MSE: ({mse_lower}, {mse_upper})")
print(f"{confidence_level*100}% confidence interval for R^2: ({r2_lower}, {r2_upper})")

with open('bootstrap_test_results.txt', 'w') as file:
    file.write(
        f"Feature confidence intervals: {feature_confidence_intervals}\n")
    file.write(
        f"Residuals confidence interval: ({residuals_lower}, {residuals_upper})\n")
    file.write(f"MSE confidence interval: ({mse_lower}, {mse_upper})\n")
    file.write(f"R2 confidence interval: ({r2_lower}, {r2_upper})\n")
    file.write(
        f"Linear residuals confidence interval: ({linear_residuals_lower}, {linear_residuals_upper})\n")
    file.write(
        f"Linear MSE confidence interval: ({linear_mse_lower}, {linear_mse_upper})\n")
    file.write(
        f"Linear R2 confidence interval: ({linear_r2_lower}, {linear_r2_upper})\n")
    file.write(
        f"Naive residuals confidence interval: ({naive_residuals_lower}, {naive_residuals_upper})\n")
    file.write(
        f"Naive MSE confidence interval: ({naive_mse_lower}, {naive_mse_upper})\n")
    file.write(
        f"Naive R2 confidence interval: ({naive_r2_lower}, {naive_r2_upper})\n")
    file.close()
