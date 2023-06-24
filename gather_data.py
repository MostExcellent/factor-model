# TODO: try monthly data and test different time periods

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
HIST_FIELDS = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
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
        # type: ignore
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
    request.append(SECURITIES, index)
    request.append(FIELDS, "INDX_MEMBERS")

    overrides = request.getElement(OVERRIDES)
    override1 = overrides.appendElement()
    override1.setElement(FIELDID, 'REFERENCE_DATE')
    override1.setElement(VALUE, f"{year}0105")

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


def fetch_projections(years, tickers_by_year):
    """
    Fetches the data for BEST_EPS and BEST_PE fields from Bloomberg for the given tickers and year.
    """
    data_rows = []

    for year in years:
        print(f"Fetching projections for {year}...")
        request = ref_data_service.createRequest("ReferenceDataRequest")
        request.append(SECURITIES, tickers_by_year[year])
        request.append(FIELDS, "BEST_EPS")
        request.append(FIELDS, "BEST_PE_RATIO")
        request.append(FIELDS, "BEST_ROE")

        overrides = request.getElement(OVERRIDES)
        override1 = overrides.appendElement()
        override1.setElement(FIELDID, 'REFERENCE_DATE')
        override1.setElement(VALUE, f"{year}0102")

        session.sendRequest(request)
        event = event_loop(session)

        for msg in event:
            security_data_array = msg.getElement('securityData')
            for securityData in security_data_array.values():
                ticker = securityData.getElementAsString('security')
                fieldData = securityData.getElement('fieldData')
                data_row = {
                    'Year': year,
                    'Ticker': ticker,
                    'ForwardEPS': fetch_field_data(fieldData, 'BEST_EPS'),
                    'ForwardPE': fetch_field_data(fieldData, 'BEST_PE_RATIO'),
                    'ForwardROE': fetch_field_data(fieldData, 'BEST_ROE')
                }
                data_rows.append(data_row)

    return pd.DataFrame.from_records(data_rows)


def get_historical_data(fields, years, members_by_year):
    """
    Gets historical data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting historical data from Bloomberg...")
    data_rows = np.ndarray((0, 5))
    for year in years:
        print(f"Year: {year}")
        tickers = members_by_year[year]
        #print(tickers)
        try:
            request = ref_data_service.createRequest(
                "HistoricalDataRequest")
            #request.set(PERIODICITY_ADJUSTMENT, "ACTUAL")
            request.set(PERIODICITY_SELECTION, "YEARLY")
            request.set(START_DATE, f"{year}0105")
            # changed end date to end of the year
            request.set(END_DATE, f"{year}0110")
            request.set(NON_TRADING_DAY_FILL_OPTION, "ALL_CALENDAR_DAYS")
            request.set(NON_TRADING_DAY_FILL_METHOD, "PREVIOUS_VALUE")
            for ticker in tickers:
                request.append(SECURITIES, ticker)
            for field in fields:
                request.append(FIELDS, field)

            session.sendRequest(request)

            event = event_loop(session)

            # Get the response
            for msg in event:
                # check if 'securityData' is present
                if msg.hasElement('securityData'):
                    security_data_array = msg.getElement('securityData')
                else:
                    print("No security data found")
                    continue
                print(security_data_array)
                for securityData in security_data_array.values():
                    print(securityData)
                    ticker = securityData.getElementAsString('security')
                    field_data = securityData.getElement('fieldData')

                    last_price = fetch_field_data(field_data, 'PX_LAST')
                    market_cap = fetch_field_data(
                        field_data, 'CUR_MKT_CAP')
                    book_value_per_share = fetch_field_data(
                        field_data, 'BOOK_VAL_PER_SH')
                    roe = fetch_field_data(field_data, 'RETURN_COM_EQY')
                    free_cash_flow = fetch_field_data(
                        field_data, 'CF_FREE_CASH_FLOW')
                    #industry_sector = get_industry_sector(ticker)
                    data_row = {
                        'Year': year,
                        'Ticker': ticker,
                        'LastPrice': last_price,
                        'MarketCap': market_cap,
                        'BookValuePerShare': book_value_per_share,
                        'ROE': roe,
                        'FreeCashFlow': free_cash_flow,
                        # 'IndustrySector': industry_sector,
                    }
                    data_rows.append(data_row)
                    print(data_row)

        except Exception as exception:
            print(f"Error for {year}: {exception}")
            continue

    fetched_df = pd.DataFrame.from_records(data_rows)
    print(fetched_df.head())

    return fetched_df


def get_reference_data(years, members_by_year):
    """
    Gets reference data from Bloomberg for the given tickers and years.
    """
    print("Getting reference data from Bloomberg...")
    data_rows = []
    for year in years:
        print(f"Year: {year}")
        try:
            request = ref_data_service.createRequest(
                "ReferenceDataRequest")
            for ticker in members_by_year[year]:
                request.append(SECURITIES, ticker)
            request.append(FIELDS, "BEST_EPS")
            request.append(FIELDS, "BEST_PE_RATIO")
            request.append(FIELDS, "BEST_ROE")
            overrides = request.getElement(OVERRIDES)
            override1 = overrides.appendElement()
            override1.setElement(FIELDID, 'REFERENCE_DATE')
            override1.setElement(VALUE, f"{year}0105")

            session.sendRequest(request)
            event = event_loop(session)

            best_eps = {}
            best_pe = {}
            best_roe = {}

            for msg in event:
                security_data_array = msg.getElement('securityData')
                for securityData in security_data_array.values():
                    ticker = securityData.getElementAsString('security')
                    fieldData = securityData.getElement('fieldData')
                    best_eps[ticker] = fetch_field_data(fieldData, 'BEST_EPS')
                    best_pe[ticker] = fetch_field_data(
                        fieldData, 'BEST_PE_RATIO')
                    best_roe[ticker] = fetch_field_data(fieldData, 'BEST_ROE')

            for ticker in members_by_year[year]:
                data_row = {
                    'Year': year,
                    'Ticker': ticker,
                    'ForwardEPS': best_eps[ticker],
                    'ForwardPE': best_pe[ticker],
                    'ForwardROE': best_roe[ticker],
                }
                data_rows.append(data_row)
                print(data_row)

        except Exception as exception:
            print(f"Error for {ticker} in {year}: {exception}")
            continue

    fetched_df = pd.DataFrame(data_rows)
    print(fetched_df.head())

    return fetched_df


members_by_year = get_indx_for_years(YEARS)
historical_data = get_historical_data(HIST_FIELDS, YEARS, members_by_year)
projections = fetch_projections(YEARS, members_by_year)

# merge historical data with projections
merged_df = pd.merge(historical_data, projections, on=['Year', 'Ticker'])

# save unprocessed data
merged_df.to_csv('unprocessed_data.csv', index=False)

# Interpolate missing values
merged_df = merged_df.groupby('Ticker').apply(
    lambda group: group.interpolate(method='linear'))
# Any values that are still NaN are set to the mean for that year
merged_df = merged_df.groupby('Year').transform(
    lambda group: group.fillna(group.mean()))
# Drop any remaining NaNs
merged_df.dropna(inplace=True)

# save processed data
merged_df.to_csv('processed_data.csv', index=False)
