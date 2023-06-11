# The basics
import pandas as pd
import numpy as np
# sklearn for regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# The Bloomberg API
import blpapi

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

years = range(2000, 2021)  # Sample period
data_dict = {year: [] for year in years}

for ticker in tickers:
    for year in years:
        request = ref_data_service.createRequest("ReferenceDataRequest")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "ANNUAL")
        request.set("startDate", f"{year}-01-01")
        request.set("endDate", f"{year}-12-31")
        request.set("nonTradingDayFillOption", "ALL_CALENDAR_DAYS")
        request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
        request.append("securities", ticker)
        for field in fields:
            request.append("fields", field)
    
        session.sendRequest(request)
    
        # Event Loop
        while True:
            event = session.nextEvent()
            if event.eventType() == blpapi.Event.RESPONSE or \
            event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                break

        # Get the response
        for msg in event:
            security_data_array = msg.getElement('securityData')
        
        for i in range(security_data_array.numValues()):
            # Stick the data into data_dict to be converted to a dataframe later
            security_data = security_data_array.getValueAsElement(i)
            field_exceptions = security_data.getElement('fieldExceptions')
                
            # If there are any field exceptions, skip this ticker for this year
            if field_exceptions.numValues() > 0:
                continue

            field_data = security_data.getElement('fieldData')
            last_price = field_data.getElementAsFloat('PX_LAST')
            market_cap = field_data.getElementAsFloat('CUR_MKT_CAP')
            book_value_per_share = field_data.getElementAsFloat('BOOK_VAL_PER_SH')
            roe = field_data.getElementAsFloat('RETURN_COM_EQY')
            free_cash_flow = field_data.getElementAsFloat('CF_FREE_CASH_FLOW')

            data_dict[year].append([ticker, last_price, market_cap, book_value_per_share, roe, free_cash_flow])

df = pd.DataFrame()
df['Year'] = years

# Noralization helper function
def normalize(x):
    return (x - x.mean()) / x.std()

def process_factors(df):
    # Calculate factors
    df['MarketPremium'] = df['LastPrice'].pct_change()
    df['Size'] = df['MarketCap']
    df['Value'] = df['BookValuePerShare'] / df['LastPrice']  # Book to market ratio
    df['Profitability'] = df['ROE']
    df['Investment'] = df['FreeCashFlow'] / df['MarketCap']

    # Normalize factors to have a common scale
    df['MarketPremiumNorm'] = normalize(df['MarketPremium'])
    df['SizeNorm'] = normalize(df['Size'])
    df['ValueNorm'] = normalize(df['Value'])
    df['ProfitabilityNorm'] = normalize(df['Profitability'])
    df['InvestmentNorm'] = normalize(df['Investment'])

    # Calculate score as a sum of all normalized factors (equal weight to all factors)
    df['Score'] = df[['MarketPremiumNorm', 'SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']].sum(axis=1)

    return df

# Calculate forward returns
df['ForwardReturn'] = df.groupby('Ticker')['LastPrice'].pct_change(-1)
# Drop the last row for each ticker as it will have a NaN for ForwardReturn
df.dropna(subset=['ForwardReturn'], inplace=True)
# Normalize forward returns
df['ForwardReturnNorm'] = df.groupby('Year')['ForwardReturn'].transform(normalize)

# Group by ticker and year
df_grouped = df.groupby(['Ticker', 'Year']).apply(process_factors)
df_grouped.reset_index(inplace=True, drop=True)

# Save to csv to avoid making API calls again
df_grouped.to_csv('data.csv')
# TODO: add argument to read from csv if it exists
# TODO: stick this into a notebook

# Now we do a regression

features = ['MarketPremiumNorm', 'SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm', 'Score']
target = 'ForwardReturnNorm'

x = df_grouped[features]
y = df_grouped[target]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Do the regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2}')