# TODO: try monthly data and test different time periods

import os
# For timeout
import time
from datetime import datetime as dt

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
               'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW',
               'OPER_MARGIN','PROF_MARGIN', 'IS_EPS', 'PE_RATIO', 'EPS_GROWTH','SALES_GROWTH']# add earnings forecasts
               # add debt-related fields

YEARS = np.arange(START_YEAR, END_YEAR)  # Sample period

# names for use with blpapi

SECURITIES = Name('securities')
FIELDS = Name('fields')
FIELDID = Name('fieldId')
FIELDDATA = Name('fieldData')
VALUE = Name('value')
OVERRIDES = Name('overrides')
OVERRIDE = Name('override')
START_DATE = Name('startDate')
END_DATE = Name('endDate')
PERIODICITY_SELECTION = Name('periodicitySelection')
PERIODICITY_ADJUSTMENT = Name('periodicityAdjustment')
NON_TRADING_DAY_FILL_OPTION = Name('nonTradingDayFillOption')
NON_TRADING_DAY_FILL_METHOD = Name('nonTradingDayFillMethod')


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
    override1.setElement(FIELDID, 'END_DATE_OVERRIDE')
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


def get_data(fields, start_year, end_year):
    """
    Gets historical and reference data from Bloomberg for the given tickers, fields and years.
    """
    print("Getting data from Bloomberg...")
    members = get_index_members(INDEX, end_year)
    data_rows = []
    for ticker in tqdm(members, desc="Getting data..."):
        try:
            request = ref_data_service.createRequest(
                "HistoricalDataRequest")
            request.set("periodicityAdjustment", "ACTUAL")
            request.set("periodicitySelection", "YEARLY")
            request.set("startDate", f"{start_year}0101")
            request.set("endDate", f"{end_year}0101")
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
                # print(security_data_array)
                # exit()
                field_data_array = security_data_array.getElement(FIELDDATA)
                for field_data in field_data_array.values():
                    date = field_data.getElementAsDatetime('date')
                    data_row = {'Date': date, 'Ticker': ticker}
                    for field in fields:
                        data_row[field] = fetch_field_data(
                            field_data, field)
                    data_rows.append(data_row)

        except Exception as exception:
            print(f"Error for {ticker}: {exception}")
            continue

    fetched_df = pd.DataFrame.from_records(data_rows)
    print(fetched_df.head())

    return fetched_df


if os.path.exists('unprocessed_data.csv'):
    data = pd.read_csv('unprocessed_data.csv')
else:
    # Start a Bloomberg session
    session = blpapi.Session()  # Start a Bloomberg session
    session.start()

    # Open the reference data service
    if not session.openService("//blp/refdata"):
        print("Failed to open //blp/refdata")
        session.stop()
        exit()

    ref_data_service = session.getService("//blp/refdata")
    data = get_data(FIELDS_LIST, YEARS[0], YEARS[-1])
    # save unprocessed data
    data.to_csv('unprocessed_data.csv', index=False)

# Remove duplicate years
data = data.drop_duplicates(subset=['Date', 'Ticker'], keep='last')

# Interpolate missing values
data = data.groupby('Ticker').apply(
    lambda group: group.interpolate(method='linear'))
# Any values that are still NaN are set to the mean for that year
data = data.groupby('Date').apply(
    lambda group: group.fillna(group.mean(numeric_only=True)))
# Drop any remaining NaNs
data.dropna(inplace=True)

# save processed data
data.to_csv('processed_data.csv', index=False)
