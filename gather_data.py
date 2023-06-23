# TODO: try monthly data and test different time periods

# To load the data from csv
import os
# For timeout
import time

# The Bloomberg API
import blpapi
import numpy as np
# Fun stuff
import pandas as pd
from blpapi import Name

START_YEAR = 2010
END_YEAR = 2021
INDEX = 'SPX Index'  # S&P 500

# Fields for historical data request
FIELDS_LIST = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
               'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']

# Start a Bloomberg session
session = blpapi.Session()  # Start a Bloomberg session
session.start()

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

# Bloomberg fields

YEARS = np.arange(START_YEAR, END_YEAR)  # Sample period

# names for use with blpapi

SECURITIES = Name('securities')
FIELDS = Name('fields')
FIELDID = Name('fieldId')
VALUE = Name('value')
OVERRIDES = Name('overrides')
OVERRIDE = Name('override')
START_DATE = Name('startDate')
END_DATE = Name('endDate')
PERIODICITY_SELECTION = Name('periodicitySelection')
PERIODICITY_ADJUSTMENT = Name('periodicityAdjustment')
NON_TRADING_DAY_FILL_OPTION = Name('nonTradingDayFillOption')
NON_TRADING_DAY_FILL_METHOD = Name('nonTradingDayFillMethod')
#FIELD_NAMES = [Name(field) for field in FIELDS_LIST]


def event_loop(e_session, timeout=7000):
    """
    Creates an event loop that waits for a response from the session.
    """
    event = None
    deadline = time.time() + timeout / 1000  # convert ms to seconds
    while True:
        event = e_session.nextEvent(timeout)
        # type: ignore
        # type: ignore
        if event.eventType() in [blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE]: # type: ignore
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
    request.append(SECURITIES, index)
    request.append(FIELDS, "INDX_MEMBERS")

    overrides = request.getElement(OVERRIDES)
    override1 = overrides.appendElement()
    override1.setElement(FIELDID, 'REFERENCE_DATE')
    override1.setElement(VALUE, f"{year}0102")

    session.sendRequest(request)

    members = []
    event = event_loop(session)
    for msg in event:
        security_data_array = msg.getElement('securityData')
        for security_data in security_data_array.values():
            field_data = security_data.getElement('fieldData')
            indx_members = field_data.getElement('INDX_MEMBERS')
            for indx_member in indx_members.values():
                member_string = indx_member.getElementAsString(
                    'Member Ticker and Exchange Code')
                member_string = member_string.replace(
                    " UW", " US Equity").replace(" UN", " US Equity")
                members.append(member_string)

    print(f"members: {members[:5]}...")
    return members

def get_indx_for_years(years, index=INDEX):
    """
    Gets the index members for the given years.
    """
    index_members_by_year = {}
    for year in years:
        index_members = get_index_members(index, year)
        index_members_by_year[year] = index_members
    return index_members_by_year


def get_industry_sector(ticker):
    """
    Gets the industry sector for the given ticker.
    """
    request = ref_data_service.createRequest("ReferenceDataRequest")
    request.append(SECURITIES, ticker)
    request.append(FIELDS, "INDUSTRY_SECTOR")

    session.sendRequest(request)

    event = event_loop(session)
    industry_sector = np.nan
    for msg in event:
        security_data_array = msg.getElement('securityData')
        for security_data in security_data_array.values():
            field_data = security_data.getElement('fieldData')
            if field_data.hasElement('INDUSTRY_SECTOR'):
                industry_sector = field_data.getElementAsString(
                    'INDUSTRY_SECTOR')

    return industry_sector


def fetch_projections(ticker, year):
    """
    Fetches the data for BEST_EPS and BEST_PE fields from Bloomberg for the given ticker and year.
    """
    request = ref_data_service.createRequest("ReferenceDataRequest")
    request.append(SECURITIES, ticker)
    request.append(FIELDS, "BEST_EPS")
    request.append(FIELDS, "BEST_PE_RATIO")
    request.append(FIELDS, "BEST_ROE")
    overrides = request.getElement(OVERRIDES)
    override1 = overrides.appendElement()
    override1.setElement(FIELDID, 'REFERENCE_DATE')
    override1.setElement(VALUE, f"{year}0101")

    session.sendRequest(request)
    event = event_loop(session)

    best_eps = np.nan
    best_pe = np.nan
    best_roe = np.nan

    for msg in event:
        security_data_array = msg.getElement('securityData')
        for securityData in security_data_array.values():
            fieldData = securityData.getElement('fieldData')
            best_eps = fetch_field_data(fieldData, 'BEST_EPS')
            best_pe = fetch_field_data(fieldData, 'BEST_PE_RATIO')
            best_roe = fetch_field_data(fieldData, 'BEST_ROE')

    return best_eps, best_pe, best_roe


def get_data(fields, years=YEARS, index=INDEX):
    """
    Gets data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting data from Bloomberg...")
    data_rows = []
    index_members = get_indx_for_years(years, index)
    for year in years:
        print(f"Year: {year}")
        for ticker in index_members[year]:
            #print(f'Processing {ticker}')
            try:
                request = ref_data_service.createRequest(
                    "HistoricalDataRequest")
                request.set(PERIODICITY_ADJUSTMENT, "ACTUAL")
                request.set(PERIODICITY_SELECTION, "YEARLY")
                request.set(START_DATE, f"{year}0101")
                # changed end date to end of the year
                request.set(END_DATE, f"{year}1231")
                request.set(NON_TRADING_DAY_FILL_OPTION, "ALL_CALENDAR_DAYS")
                request.set(NON_TRADING_DAY_FILL_METHOD, "PREVIOUS_VALUE")
                request.append(SECURITIES, ticker)
                for field in fields:
                    request.append(FIELDS, field)

                session.sendRequest(request)

                event = event_loop(session)

                # Get the response
                for msg in event:
                    # check if 'securityData' is present
                    if msg.hasElement('securityData'):
                        security_data = msg.getElement('securityData')
                    else:
                        continue

                    # field_exceptions = security_data.getElement(
                    #     'fieldExceptions')

                    # # If there are any field exceptions, skip this ticker for this year
                    # if field_exceptions.numValues() > 0:
                    #     continue

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
                        #industry_sector = get_industry_sector(ticker)
                        best_eps, best_pe, best_roe = fetch_projections(
                            ticker, year)
                        data_rows.append({
                            'Year': year,
                            'Ticker': ticker,
                            'LastPrice': last_price,
                            'MarketCap': market_cap,
                            'BookValuePerShare': book_value_per_share,
                            'ROE': roe,
                            'FreeCashFlow': free_cash_flow,
                            # 'IndustrySector': industry_sector,
                            'ForwardEPS': best_eps,
                            'ForwardPE': best_pe,
                            'ForwardROE': best_roe,
                        })
                        print(data_rows[-1])
            except Exception as exception:
                print(f"Error for {ticker} in {year}: {exception}")
                continue

    fetched_df = pd.DataFrame(data_rows)
    print(fetched_df.head())
    # Handle missing values by interpolation, then drop remaining NaNs

    fetched_df = df.groupby('Ticker').apply(
        lambda group: group.interpolate(method='linear'))
    # fetched_df.dropna(inplace=True)
    fetched_df.dropna(subset=['LastPrice'], inplace=True)

    # calculate the yearly mean for each column
    yearly_means = fetched_df.groupby('Year').transform('mean')

    # fill remaining NaNs with the yearly mean
    fetched_df.fillna(yearly_means, inplace=True)

    return fetched_df


# Load data from csv if it exists, else fetch from Bloomberg API
CSV_FILE = 'data.csv'
# Check if the file exists
if os.path.isfile(CSV_FILE):
    # Read from the file
    print("Reading data from csv...")
    df = pd.read_csv(CSV_FILE)
else:
    print("Fetching data from Bloomberg API...")
    # Fetch data from the Bloomberg API
    df = get_data(FIELDS_LIST, YEARS, INDEX)
    # Save to csv to avoid making API calls again in the future
    df.to_csv(CSV_FILE, index=False)
