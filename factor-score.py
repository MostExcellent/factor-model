import pandas as pd
import blpapi  # The Bloomberg API
from datetime import datetime, timedelta
import time
import numpy as np

session = blpapi.Session()  # Start a Bloomberg session
session.start()

INDEX = "SPX Index"  # S&P 500 Index

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH', 'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']  # Bloomberg fields

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
                member_string = indx_member.getElementAsString('Member Ticker and Exchange Code')
                member_string = member_string.replace(" UW", " US Equity").replace(" UN", " US Equity")
                members.append(member_string)

    print(f"members: {members[:5]}...")
    return members

def get_data(fields, index=INDEX, tickers=None, merge_tickers_index=False):
    """
    Gets data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting data from Bloomberg...")
    data_rows = []

    # Get today's date and the date one year ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Format the dates as strings
    end_date = end_date.strftime('%Y%m%d')
    start_date = start_date.strftime('%Y%m%d')

    # Get the index members for the current day
    if tickers is None:
        members = get_index_members(index)
    elif merge_tickers_index:
        members = tickers + get_index_members(index)
        members = list(set(members))
    else:
        members = tickers

    for ticker in members:
        for date in [start_date, end_date]:
            try:
                request = ref_data_service.createRequest(
                    "HistoricalDataRequest")
                request.set("periodicityAdjustment", "ACTUAL")
                request.set("periodicitySelection", "YEARLY")
                request.set("startDate", date)
                request.set("endDate", date)
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

                    # If there are any field exceptions, skip this ticker
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
                        industry_sector = fetch_field_data(field_data, 'INDUSTRY_SECTOR')

                        data_rows.append({
                            'Date': field_data.getElementAsDatetime('date').date(),
                            'Ticker': ticker,
                            'LastPrice': last_price,
                            'MarketCap': market_cap,
                            'BookValuePerShare': book_value_per_share,
                            'ROE': roe,
                            'FreeCashFlow': free_cash_flow,
                            'IndustrySector': industry_sector,
                        })
            except Exception as e:
                print(f"Error for {ticker} on {date}: {e}")

    df = pd.DataFrame(data_rows)

    # Handle missing values by interpolation, then drop remaining NaNs
    df = df.groupby('Ticker').apply(
        lambda group: group.interpolate(method='linear'))
    df.dropna(inplace=True)

    return df

df = get_data(fields, index)  # Get data for all tickers and fields

# Calculate factors and normalize as you've done previously
df['Size'] = df['MarketCap']
df['Value'] = df['BookValuePerShare'] / df['LastPrice']  # Book to market ratio
df['Profitability'] = df['ROE']
df['Investment'] = df['FreeCashFlow'] / df['MarketCap']

# Normalize factors to have a common scale
df['SizeNorm'] = (df['Size'] - df['Size'].mean()) / df['Size'].std()
df['ValueNorm'] = (df['Value'] - df['Value'].mean()) / df['Value'].std()
df['ProfitabilityNorm'] = (df['Profitability'] - df['Profitability'].mean()) / df['Profitability'].std()
df['InvestmentNorm'] = (df['Investment'] - df['Investment'].mean()) / df['Investment'].std()

# Calculate score as a sum of all normalized factors (equal weight to all factors)
#df['Score'] = df[['SizeNorm', 'ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']].sum(axis=1)
df['Score'] = df[['ValueNorm', 'ProfitabilityNorm', 'InvestmentNorm']].mean(axis=1)

# Sort dataframe by score
df_sorted = df.sort_values('Score', ascending=False)

print(df_sorted)

df_sorted.to_excel('factor-scores.xlsx')