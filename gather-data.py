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

fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
          'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']  # Bloomberg fields

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


def fetch_field_data(field_data, field_name):
    """
    Fetches the data for a specific field from the field data.
    """
    if field_data.hasElement(field_name):
        return field_data.getElementAsFloat(field_name)
    else:
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
                member_string = indx_member.getElementAsString(
                    'Member Ticker and Exchange Code')
                member_string = member_string.replace(
                    " UW", " US Equity").replace(" UN", " US Equity")
                members.append(member_string)

    print(f"members: {members[:5]}...")
    return members


def get_industry_sector(ticker):
    """
    Gets the industry sector for the given ticker.
    """
    request = ref_data_service.createRequest("ReferenceDataRequest")
    request.append("securities", ticker)
    request.append("fields", "INDUSTRY_SECTOR")

    session.sendRequest(request)

    event = event_loop(session)
    for msg in event:
        securityDataArray = msg.getElement('securityData')
        for securityData in securityDataArray.values():
            fieldData = securityData.getElement('fieldData')
            if fieldData.hasElement('INDUSTRY_SECTOR'):
                industry_sector = fieldData.getElementAsString(
                    'INDUSTRY_SECTOR')
            else:
                industry_sector = np.nan

    return industry_sector


def get_data(fields, years, index=INDEX):
    """
    Gets data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting data from Bloomberg...")
    data_rows = []
    for year in years:
        print(f"Year: {year}")
        for ticker in get_index_members(index, year):
            #print(f'Processing {ticker}')
            try:
                request = ref_data_service.createRequest(
                    "HistoricalDataRequest")
                request.set("periodicityAdjustment", "ACTUAL")
                request.set("periodicitySelection", "YEARLY")
                request.set("startDate", f"{year}0101")
                # changed end date to end of the year
                request.set("endDate", f"{year}1231")
                request.set("nonTradingDayFillOption", "ALL_CALENDAR_DAYS")
                request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
                request.append("securities", ticker)
                for field in fields:
                    request.append("fields", field)

                session.sendRequest(request)

                event = event_loop(session)

                # Get the response
                for msg in event:
                    # check if 'securityData' is present
                    if msg.hasElement('securityData'):
                        security_data = msg.getElement('securityData')
                    else:
                        continue

                    field_exceptions = security_data.getElement(
                        'fieldExceptions')

                    # If there are any field exceptions, skip this ticker for this year
                    if field_exceptions.numValues() > 0:
                        continue

                    field_data_array = security_data.getElement('fieldData')

                    for j in range(field_data_array.numValues()):
                        field_data = field_data_array.getValueAsElement(j)

                        last_price = fetch_field_data(field_data, 'PX_LAST')
                        market_cap = fetch_field_data(
                            field_data, 'CUR_MKT_CAP')
                        book_value_per_share = fetch_field_data(
                            field_data, 'BOOK_VAL_PER_SH')
                        roe = fetch_field_data(field_data, 'RETURN_COM_EQY')
                        free_cash_flow = fetch_field_data(
                            field_data, 'CF_FREE_CASH_FLOW')
                        industry_sector = get_industry_sector(ticker)

                        data_rows.append({
                            'Year': year,
                            'Ticker': ticker,
                            'LastPrice': last_price,
                            'MarketCap': market_cap,
                            'BookValuePerShare': book_value_per_share,
                            'ROE': roe,
                            'FreeCashFlow': free_cash_flow,
                            'IndustrySector': industry_sector,
                        })
                        print(data_rows)
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
                    'IndustrySector': np.nan,
                })

    df = pd.DataFrame(data_rows)
    print(df)

    # Handle missing values by interpolation, then drop remaining NaNs
    print(df.head())
    df = df.groupby('Ticker').apply(
        lambda group: group.interpolate(method='linear'))
    df.dropna(inplace=True)

    return df


def get_risk_free_rate(years=years):
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
    # Remember, we want the average for the year
    request.set("endDate", f"{years[-1]}1231")
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

# If risk free rates csv exists, read from it, else fetch from Bloomberg API
csv_risk_free_rates = 'risk_free_rates.csv'

try:
    print("Reading risk free rates from csv...")
    df_rates = pd.read_csv(csv_risk_free_rates)
    risk_free_rates = df_rates.set_index('Year')['Rate'].to_dict()
except FileNotFoundError:
    print("File not found, fetching risk free rates from Bloomberg API")
    risk_free_rates = get_risk_free_rate()
    # Save risk_free_rates into a CSV file
    pd.DataFrame(list(risk_free_rates.items()), columns=[
                 'Year', 'Rate']).to_csv(csv_risk_free_rates, index=False)

#df['RiskFreeRate'] = df['Year'].map(risk_free_rates)
