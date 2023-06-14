# TODO: try monthly data and test different time periods

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
# To load the data from csv
import os
# For timeout
import time

START_YEAR = 2010
END_YEAR = 2021
INDEX = 'SPX Index'  # S&P 500

# Start a Bloomberg session
session = blpapi.Session()  # Start a Bloomberg session
session.start()

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH', 'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']  # Bloomberg fields

years = np.arange(START_YEAR, END_YEAR)  # Sample period

def event_loop(session, timeout=7000):
    """
    Creates an event loop that waits for a response from the session.
    """
    event = None
    deadline = time.time() + timeout / 1000  # convert ms to seconds
    while True:
        event = session.nextEvent(timeout)
        if event.eventType() in [blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE]:
            break
        if time.time() > deadline:
            break
    return event

# TODO: separate the code for making a request.

def fetch_field_data(field_data, field):
    try:
        return field_data.getElementAsFloat(field)
    except blpapi.NotFoundException:
        return np.nan

def get_index_members(index, year):
    """
    Gets the index members for the given index and year.
    """
    request = ref_data_service.createRequest("ReferenceDataRequest")
    request.append("securities", index)
    request.append("fields", "INDX_MEMBERS")

    overrides = request.getElement('overrides')
    override1 = overrides.appendElement()
    override1.setElement('fieldId', 'REFERENCE_DATE')
    override1.setElement('value', f"{year}1231")

    session.sendRequest(request)

    members = []
    event = event_loop(session)
    for msg in event:
        securityDataArray = msg.getElement('securityData')
        for securityData in securityDataArray.values():
            fieldData = securityData.getElement('fieldData')
            indx_members = fieldData.getElement('INDX_MEMBERS')
            for indx_member in indx_members.values():
                member_string = indx_member.getElementAsString('Member Ticker and Exchange Code')
                member_string = member_string.replace(" UW", " US Equity").replace(" UN", " US Equity")
                members.append(member_string)

    print(f"members: {members[:5]}...")
    return members

def get_data(fields, years, index=INDEX):
    """
    Gets data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting data from Bloomberg...")
    data_rows = []
    for year in years:
        for ticker in get_index_members(index, year):
            #print(f'Processing {ticker}')
            try:
                request = ref_data_service.createRequest("HistoricalDataRequest")
                request.set("periodicityAdjustment", "ACTUAL")
                request.set("periodicitySelection", "YEARLY")
                request.set("startDate", f"{year}0101")
                request.set("endDate", f"{year}1231") # changed end date to end of the year
                request.set("nonTradingDayFillOption", "ALL_CALENDAR_DAYS")
                request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
                request.append("securities", ticker)
                for field in fields:
                    request.append("fields", field)
            
                session.sendRequest(request)

                event = event_loop(session)

                # Get the response
                for msg in event:
                    #print(msg)
                    if msg.hasElement('securityData'): # check if 'securityData' is present
                        security_data_array = msg.getElement('securityData')
                    else:
                        continue
            
                    for i in range(security_data_array.numValues()):
                        security_data = security_data_array.getValueAsElement(i)
                        field_exceptions = security_data.getElement('fieldExceptions')
                    
                        # If there are any field exceptions, skip this ticker for this year
                        if field_exceptions.numValues() > 0:
                            continue

                        field_data = security_data.getElement('fieldData')
                
                        last_price = fetch_field_data(field_data, 'PX_LAST')
                        print(f'Last price for {ticker}: {last_price}')
                        market_cap = fetch_field_data(field_data, 'CUR_MKT_CAP')
                        book_value_per_share = fetch_field_data(field_data, 'BOOK_VAL_PER_SH')
                        roe = fetch_field_data(field_data, 'RETURN_COM_EQY')
                        free_cash_flow = fetch_field_data(field_data, 'CF_FREE_CASH_FLOW')

                        data_rows.append({
                            'Year': year,
                            'Ticker': ticker,
                            'LastPrice': last_price,
                            'MarketCap': market_cap,
                            'BookValuePerShare': book_value_per_share,
                            'ROE': roe,
                            'FreeCashFlow': free_cash_flow,
                        })

            except Exception as e:
                print(f"Error for {ticker} in {year}: {e}")
                # Append a placeholder row with NaNs in case there are issues. (for delisted stocks?)
                data_rows.append({
                    'Year': year,
                    'Ticker': ticker,
                    'LastPrice': np.nan,
                    'MarketCap': np.nan,
                    'BookValuePerShare': np.nan,
                    'ROE': np.nan,
                    'FreeCashFlow': np.nan,
                })
                
    df = pd.DataFrame(data_rows)
    print(df)

    # Handle missing values by interpolation, then drop remaining NaNs
    print(df.head())
    df = df.groupby('Ticker').apply(lambda group: group.interpolate(method='linear'))
    df.dropna(inplace=True)

    return df


def get_risk_free_rate(years = years):
    """
    Returns average risk free rate for each year in years as a dictionary
    """
    print("Getting risk free rates...")
    risk_free_rates = {}

    request = ref_data_service.createRequest("HistoricalDataRequest")
    request.set("securities", "USGG10YR Index")
    request.set("fields", "PX_LAST")
    request.set("periodicityAdjustment", "MONTHLY")
    request.set("periodicitySelection", "MONTHLY")
    request.set("startDate", f"{years[0]}0101")
    request.set("endDate", f"{years[-1]}1231") # Remember, we want the average for the year
    request.set("nonTradingDayFillOption", "ALL_CALENDAR_DAYS")
    request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")

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
    print("Processing factors...")
    df_copy = df.copy()
    df_copy['MarketPremium'] = df_copy.groupby('Ticker')['LastPrice'].pct_change() - df_copy['RiskFreeRate']
     # Size premium is incorrect, and is a plaholder
    df_copy['Size'] = df_copy['MarketCap'] # This is still useful for the regressor, however
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

def train_model(x_train, y_train):
    print("Training model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    return rf

def test_model(rf, x_test, y_test):
    print("Testing model...")
    y_pred = rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

# Load data from csv if it exists, else fetch from Bloomberg API
csv_file = 'data.csv'
# Check if the file exists
if os.path.isfile(csv_file):
    # Read from the file
    print("Reading data from csv...")
    df = pd.read_csv(csv_file)
else:
    print("Fetching data from Bloomberg API...")
    # Fetch data from the Bloomberg API
    df = get_data(fields, years)
    # Save to csv to avoid making API calls again in the future
    df.to_csv(csv_file, index=False)


# Cap and floor outliers
df = cap_and_floor(df, 'LastPrice', 0.01, 0.99)
df = cap_and_floor(df, 'MarketCap', 0.01, 0.99)
df = cap_and_floor(df, 'BookValuePerShare', 0.01, 0.99)
df = cap_and_floor(df, 'ROE', 0.01, 0.99)
df = cap_and_floor(df, 'FreeCashFlow', 0.01, 0.99)

# Validate data types
print("Validating data types...")
assert df['Year'].dtype == np.int64, "Year should be an integer"
assert df['Ticker'].dtype == str, "Ticker should be a string"
assert df['LastPrice'].dtype == np.float64, "LastPrice should be a float"
assert df['MarketCap'].dtype == np.float64, "MarketCap should be a float"
assert df['BookValuePerShare'].dtype == np.float64, "BookValuePerShare should be a float"
assert df['ROE'].dtype == np.float64, "ROE should be a float"
assert df['FreeCashFlow'].dtype == np.float64, "FreeCashFlow should be a float"

# Check for duplicates
assert df.duplicated().sum() == 0, "Data contains duplicated rows"

if df.isnull().any().any():
    print("Warning: The data contains missing values")

# If risk free rates csv exists, read from it, else fetch from Bloomberg API
csv_risk_free_rates = 'risk_free_rates.csv'

try:
    print("Reading risk free rates from csv...")
    df_rates = pd.read_csv(csv_risk_free_rates)
    risk_free_rates = df_rates.groupby('Year')['Rate'].mean().to_dict()
except FileNotFoundError:
    print("File not found, fetching risk free rates from Bloomberg API")
    risk_free_rates = get_risk_free_rate()

df['RiskFreeRate'] = df['Year'].map(risk_free_rates)

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
csv_processed = 'processed_data.csv'
df_grouped.to_csv(csv_processed, index=False)
# TODO: stick this into a notebook

# Now we do a regression

features = ['MarketPremiumNorm', 'SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm'] #,'Score']
target = 'ForwardReturnNorm'

x = df_grouped[features]
y = df_grouped[target]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and fit model
rf = train_model(x_train, y_train)

# Test model
y_pred, mse, r2 = test_model(rf, x_test, y_test)

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

# Bootstrap analysis
n_samples = 1000

residuals = []
feature_importances = []
mse_vals = []
r2_vals = []

for _ in range(n_samples):
    # Bootstrap sample (with replacement)
    sample_df = df_grouped.sample(frac=1, replace=True, random_state=42)
    x_sample = sample_df[features]
    y_sample = sample_df[target]

    # Split data into train and test sets
    x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(x_sample, y_sample, test_size=0.2, random_state=42)

    # Train and test model on the bootstrap sample
    rf = train_model(x_train_sample, y_train_sample)
    y_pred, mse, r2 = test_model(rf, x_test_sample, y_test_sample)

    # Record the results
    feature_importances.append(rf.feature_importances_)
    residuals.append(y_test_sample - y_pred)
    mse_vals.append(mse)
    r2_vals.append(r2)

# Now we can calculate the confidence intervals
confidence_level = 0.95

feature_importances_transposed = list(map(list, zip(*feature_importances)))
feature_confidence_intervals = []

for feature_importances in feature_importances_transposed:
    lower = np.percentile(feature_importances, ((1 - confidence_level) / 2) * 100)
    upper = np.percentile(feature_importances, (confidence_level + ((1 - confidence_level) / 2)) * 100)
    feature_confidence_intervals.append((lower, upper))

residuals_lower = np.percentile(residuals, ((1 - confidence_level) / 2) * 100)
residuals_upper = np.percentile(residuals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
mse_lower = np.percentile(mse_vals, ((1 - confidence_level) / 2) * 100)
mse_upper = np.percentile(mse_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)
r2_lower = np.percentile(r2_vals, ((1 - confidence_level) / 2) * 100)
r2_upper = np.percentile(r2_vals, (confidence_level + ((1 - confidence_level) / 2)) * 100)

for i, (lower, upper) in enumerate(feature_confidence_intervals):
    print(f"{confidence_level*100}% confidence interval for feature {i}'s importance: ({lower}, {upper})")

print(f"{confidence_level*100}% confidence interval for residuals: ({residuals_lower}, {residuals_upper})")
print(f"{confidence_level*100}% confidence interval for MSE: ({mse_lower}, {mse_upper})")
print(f"{confidence_level*100}% confidence interval for R^2: ({r2_lower}, {r2_upper})")