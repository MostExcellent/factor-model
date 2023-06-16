import pandas as pd
import blpapi  # The Bloomberg API
from datetime import datetime, timedelta
import time
import numpy as np
import os

session = blpapi.Session()  # Start a Bloomberg session
session.start()

tickers_file = 'tickers.txt'
if os.path.exists(tickers_file):
    tickers = []
    with open(tickers_file, 'r') as file:
        for line in file:
            ticker = line.strip()  # Remove leading/trailing whitespace, including newline characters
            tickers.append(ticker+ " US Equity")
else:
    tickers = []  # Empty list if the file doesn't exist

INDEX = "SPX Index"  # S&P 500 Index

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
          'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']  # Bloomberg fields

data = []
current_year = datetime.now().year
index = "SPX Index"  # S&P 500 Index


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


def get_index_members(index):
    """
    Gets the current index members for the given index.
    """
    request = ref_data_service.createRequest("ReferenceDataRequest")
    request.append("securities", index)
    request.append("fields", "INDX_MEMBERS")
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

    return members


def get_previous_year_data(ticker, year):
    """
    Gets the last year's price for the given ticker and year.
    """
    request = ref_data_service.createRequest("HistoricalDataRequest")
    request.set("periodicityAdjustment", "ACTUAL")
    request.set("periodicitySelection", "YEARLY")
    request.set("startDate", f"{year-1}0101")
    request.set("endDate", f"{year-1}1231")
    request.set("nonTradingDayFillOption", "ALL_CALENDAR_DAYS")
    request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")
    request.append("securities", ticker)
    request.append("fields", "PX_LAST")

    session.sendRequest(request)

    event = event_loop(session)

    # Get the response
    last_price = np.nan
    for msg in event:
        if msg.hasElement('securityData'):
            security_data = msg.getElement('securityData')
        else:
            continue

        field_exceptions = security_data.getElement('fieldExceptions')

        if field_exceptions.numValues() > 0:
            continue

        field_data_array = security_data.getElement('fieldData')
        if field_data_array.numValues() > 0:
            field_data = field_data_array.getValueAsElement(0)
            last_price = fetch_field_data(field_data, 'PX_LAST')

    return last_price

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
                industry_sector = fieldData.getElementAsString('INDUSTRY_SECTOR')
            else:
                industry_sector = np.nan

    return industry_sector

def get_current_data(tickers):
    """
    Gets current data for the given tickers.
    """
    fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
              'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW', 'INDUSTRY_SECTOR']

    data_rows = []
    for ticker in tickers:
        request = ref_data_service.createRequest("ReferenceDataRequest")
        request.append("securities", ticker)
        for field in fields:
            request.append("fields", field)
        session.sendRequest(request)

        event = event_loop(session)

        for msg in event:
            if msg.hasElement('securityData'):
                security_data = msg.getElement('securityData')
            else:
                continue

            field_exceptions = security_data.getElement('fieldExceptions')
            if field_exceptions.numValues() > 0:
                continue

            field_data = security_data.getElement('fieldData')
            last_price = fetch_field_data(field_data, 'PX_LAST')
            market_cap = fetch_field_data(field_data, 'CUR_MKT_CAP')
            book_value_per_share = fetch_field_data(
                field_data, 'BOOK_VAL_PER_SH')
            roe = fetch_field_data(field_data, 'RETURN_COM_EQY')
            free_cash_flow = fetch_field_data(field_data, 'CF_FREE_CASH_FLOW')
            industry_sector = get_industry_sector(ticker)

            data_rows.append({
                'Ticker': ticker,
                'LastPrice': last_price,
                'MarketCap': market_cap,
                'BookValuePerShare': book_value_per_share,
                'ROE': roe,
                'FreeCashFlow': free_cash_flow,
                'IndustrySector': industry_sector,
            })

    df = pd.DataFrame(data_rows)

    # Handle missing values by interpolation, then drop remaining NaNs
    df = df.groupby('Ticker').apply(
        lambda group: group.interpolate(method='linear'))
    df.dropna(inplace=True)

    return df

to_get = tickers + get_index_members(INDEX)

# Get data for all tickers and fields
df = get_current_data(list(set(to_get)))

df.to_csv('current_data.csv', index=False)  # Save data to CSV file