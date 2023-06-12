# The basics
import pandas as pd
import numpy as np
# sklearn for regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
# The Bloomberg API
import blpapi

START_YEAR = 2010
END_YEAR = 2021

# Start a Bloomberg session
session = blpapi.Session()  # Start a Bloomberg session
session.start()

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

tickers = ['AAPL US Equity', 'GOOGL US Equity', 'MSFT US Equity', 'AMZN US Equity', 'META US Equity']  # Bloomberg format for tickers
fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH', 'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']  # Bloomberg fields

years = np.arange(START_YEAR, END_YEAR)  # Sample period

def event_loop(session):
    """
    An event loop to wait for the response from the Bloomberg API.
    """
    while True:
        event = session.nextEvent()
        if event.eventType() == blpapi.Event.RESPONSE or \
        event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
            break
    return event

def fetch_field_data(field_data, field):
    try:
        return field_data.getElementAsFloat(field)
    except blpapi.NotFoundException:
        return np.nan

def get_data(tickers, fields, years):
    """
    Gets data from Bloomberg for the given tickers, fields and years.
    """
    data_rows = []
    for ticker in tickers:
        request = ref_data_service.createRequest("HistoricalDataRequest")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "YEARLY")
        request.set("startDate", f"{years[0]}-01-01")
        request.set("endDate", f"{years[-1]}-12-31")
        request.set("nonTradingDayFillOption", "ALL_CALENDAR_DAYS")
        request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
        request.append("securities", ticker)
        for field in fields:
            request.append("fields", field)
    
        session.sendRequest(request)

        event = event_loop(session)

        # Get the response
        for msg in event:
            security_data_array = msg.getElement('securityData')
        
        for i in range(security_data_array.numValues()):
            security_data = security_data_array.getValueAsElement(i)
            field_exceptions = security_data.getElement('fieldExceptions')
                
            # If there are any field exceptions, skip this ticker for this year
            if field_exceptions.numValues() > 0:
                continue

            field_data = security_data.getElement('fieldData')
            
            for y in years:
                last_price = fetch_field_data(field_data, 'PX_LAST')
                market_cap = fetch_field_data(field_data, 'CUR_MKT_CAP')
                book_value_per_share = fetch_field_data(field_data, 'BOOK_VAL_PER_SH')
                roe = fetch_field_data(field_data, 'RETURN_COM_EQY')
                free_cash_flow = fetch_field_data(field_data, 'CF_FREE_CASH_FLOW')

                data_rows.append({
                    'Year': y,
                    'Ticker': ticker,
                    'LastPrice': last_price,
                    'MarketCap': market_cap,
                    'BookValuePerShare': book_value_per_share,
                    'ROE': roe,
                    'FreeCashFlow': free_cash_flow,
                })

    return pd.DataFrame(data_rows)

def get_risk_free_rate(years = years):
    """
    Returns average risk free rate for each year in years as a dictionary
    """
    risk_free_rates = {}

    request = ref_data_service.createRequest("HistoricalDataRequest")
    request.set("securities", "USGG10YR Index")
    request.set("fields", "PX_LAST")
    request.set("periodicityAdjustment", "MONTHLY")
    request.set("periodicitySelection", "MONTHLY")
    request.set("startDate", f"{years[0]}-01-01")
    request.set("endDate", f"{years[-1]}-12-31")

    event = event_loop(session)
    
    for msg in event:
        security_data = msg.getElement('securityData')
        field_data = security_data.getElement('fieldData')

        rate_data = []
        for i in range(field_data.numValues()):
            data = field_data.getValueAsElement(i)
            date = data.getElementAsDatetime("date")
            rate = data.getElementAsFloat("PX_LAST")
            rate_data.append([date.year(), rate])

        df_rates = pd.DataFrame(rate_data, columns=['Year', 'Rate'])
        risk_free_rates = df_rates.groupby('Year').mean().to_dict()['Rate']
    
    return risk_free_rates

# Noralization helper function
def normalize(x):
    return (x - x.mean()) / x.std()

# Cap and floor outliers
def cap_and_floor(df, column, lower_percentile, upper_percentile):
    lower, upper = df[column].quantile([lower_percentile, upper_percentile])
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    return df

def process_factors(df):
    """
    Calculate factors and their normalized versions.
    """
    df_copy = df.copy()
    df_copy['MarketPremium'] = df_copy.groupby('Ticker')['LastPrice'].pct_change() - df_copy['RiskFreeRate']
    df_copy['Size'] = df_copy['MarketCap'] # This is incorrect, and is a plaholder until I implement CAPM
    df_copy['Value'] = df_copy['BookValuePerShare'] / df_copy['LastPrice']  
    df_copy['Profitability'] = df_copy['ROE']
    df_copy['Investment'] = df_copy['FreeCashFlow'] / df_copy['MarketCap']

    # Normalize factors to have a common scale
    df_copy['MarketPremiumNorm'] = normalize(df_copy['MarketPremium'])
    df_copy['SizeNorm'] = normalize(df_copy['Size'])
    df_copy['ValueNorm'] = normalize(df_copy['Value'])
    df_copy['ProfitabilityNorm'] = normalize(df_copy['Profitability'])
    df_copy['InvestmentNorm'] = normalize(df_copy['Investment'])

    # Calculate score as a sum of all normalized factors (equal weight to all factors)
    df_copy['Score'] = df_copy[['MarketPremiumNorm', 'SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']].sum(axis=1)

    return df_copy

# Get data
df = get_data(tickers, fields, years)

# Cap and floor outliers
df = cap_and_floor(df, 'LastPrice', 0.01, 0.99)
df = cap_and_floor(df, 'MarketCap', 0.01, 0.99)
df = cap_and_floor(df, 'BookValuePerShare', 0.01, 0.99)
df = cap_and_floor(df, 'ROE', 0.01, 0.99)
df = cap_and_floor(df, 'FreeCashFlow', 0.01, 0.99)

# Validate data types
assert df['Year'].dtype == np.int64, "Year should be an integer"
assert df['Ticker'].dtype == str, "Ticker should be a string"
assert df['LastPrice'].dtype == np.float64, "LastPrice should be a float"
assert df['MarketCap'].dtype == np.float64, "MarketCap should be a float"
assert df['BookValuePerShare'].dtype == np.float64, "BookValuePerShare should be a float"
assert df['ROE'].dtype == np.float64, "ROE should be a float"
assert df['FreeCashFlow'].dtype == np.float64, "FreeCashFlow should be a float"

# Validate tickers
assert set(df['Ticker']).issubset(set(tickers)), "Unexpected ticker symbols in the data"

# Check for duplicates
assert df.duplicated().sum() == 0, "Data contains duplicated rows"

if df.isnull().any().any():
    print("Warning: The data contains missing values")

df['RiskFreeRate'] = df['Year'].map(get_risk_free_rate())

# Calculate forward returns
df['ForwardReturn'] = df.groupby('Ticker')['LastPrice'].pct_change(-1)
# Drop the last row for each ticker as it will have a NaN for ForwardReturn
df.dropna(subset=['ForwardReturn'], inplace=True)
# Normalize forward returns
df['ForwardReturnNorm'] = df.groupby('Year')['ForwardReturn'].transform(normalize)

# Group by ticker and year and calculate factors
df_grouped = df.groupby(['Ticker', 'Year']).apply(process_factors)
df_grouped.reset_index(inplace=True, drop=True)

# Save to csv to avoid making API calls again in the future
df_grouped.to_csv('data.csv')
# TODO: add argument to read from csv if it exists
# TODO: stick this into a notebook

# Now we do a regression

features = ['MarketPremiumNorm', 'SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm'] #,'Score']
target = 'ForwardReturnNorm'

x = df_grouped[features]
y = df_grouped[target]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and fit model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Feature importance: {rf.feature_importances_}')
print(f'MSE: {mse}')
print(f'R-squared: {r2}')

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
for factor in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_grouped[factor], bins='auto', kde=True)
    plt.title(f'Distribution of {factor}')
    plt.savefig(f'{factor}_distribution.png')

# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_grouped[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')

# Scatter plot of predicted vs. actual returns
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title('Predicted vs. Actual Returns')
plt.savefig('predicted_vs_actual.png')