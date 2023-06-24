# TODO: try monthly data and test different time periods

# For timeout
import time

# The Bloomberg API
import blpapi
import numpy as np
# Fun stuff
import pandas as pd
from blpapi import Name
from tqdm import tqdm

START_YEAR = 2010
END_YEAR = 2021
INDEX = 'SPX Index'  # S&P 500

# Fields for historical data request
FIELDS_LIST = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
               'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW', 'BEST_EPS', 'BEST_PE_RATIO', 'BEST_ROE', 'INDUSTRY_SECTOR']

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


def get_data(fields, years):
    """
    Gets historical and reference data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting data from Bloomberg...")
    print("Getting index members...")
    members_by_year = get_indx_for_years(years)
    print("Getting field data...")
    data_rows = []
    for year in tqdm(years, desc="Years"):
        tickers = members_by_year[year]
        try:
            for ticker in tqdm(tickers, desc="Tickers"):
                request = ref_data_service.createRequest(
                    "ReferenceDataRequest")
                request.append(SECURITIES, ticker)
                for field in fields:
                    request.append(FIELDS, field)

                overrides = request.getElement(OVERRIDES)
                override1 = overrides.appendElement()
                override1.setElement(FIELDID, 'REFERENCE_DATE')
                override1.setElement(VALUE, f"{year}0105")

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
                    for securityData in security_data_array.values():
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
                        best_eps = fetch_field_data(field_data, 'BEST_EPS')
                        best_pe = fetch_field_data(field_data, 'BEST_PE_RATIO')
                        best_roe = fetch_field_data(field_data, 'BEST_ROE')

                        data_row = {
                            'Year': year,
                            'Ticker': ticker,
                            'LastPrice': last_price,
                            'MarketCap': market_cap,
                            'BookValuePerShare': book_value_per_share,
                            'ROE': roe,
                            'FreeCashFlow': free_cash_flow,
                            'ForwardEPS': best_eps,
                            'ForwardPE': best_pe,
                            'ForwardROE': best_roe,
                        }
                        data_rows.append(data_row)
                        print(data_row)

        except Exception as exception:
            print(f"Error for {year}: {exception}")
            continue

    fetched_df = pd.DataFrame.from_records(data_rows)
    print(fetched_df.head())

    return fetched_df


members_by_year = get_indx_for_years(YEARS)
data = get_data(FIELDS, YEARS)

# save unprocessed data
data.to_csv('unprocessed_data.csv', index=False)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Interpolate missing values
data = data.groupby('Ticker').apply(
    lambda group: group.interpolate(method='linear'))
# Any values that are still NaN are set to the mean for that year
data = data.groupby('Year').transform(
    lambda group: group.fillna(group.mean()))
# Drop any remaining NaNs
data.dropna(inplace=True)

# save processed data
data.to_csv('processed_data.csv', index=False)
