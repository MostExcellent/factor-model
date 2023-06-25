import argparse
import os
import time
from datetime import datetime

import blpapi  # The Bloomberg API
import numpy as np
import pandas as pd
from blpapi import Name
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

INDEX = "SPX Index"  # S&P 500 Index
FIELDS_LIST = ['PX_LAST', 'HIST_TRR_PREV_1YR', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH',
               'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW',
               'OPER_MARGIN', 'PROF_MARGIN', 'IS_EPS', 'PE_RATIO', 'EPS_GROWTH', 'SALES_GROWTH']

parser = argparse.ArgumentParser()
parser.add_argument(
    "--date", help="The date to get the current data from in YYYYMMDD format. Default is current date.")
args = parser.parse_args()

if args.date:
    current_date = datetime.strptime(args.date, '%Y%m%d')
else:
    current_date = datetime.now()

session = blpapi.Session()  # Start a Bloomberg session
session.start()

TICKERS_FILE = 'tickers.txt'
if os.path.exists(TICKERS_FILE):
    tickers = []
    with open(TICKERS_FILE, 'r') as file:
        for line in file:
            ticker = line.strip()  # Remove leading/trailing whitespace, including newline characters
            tickers.append(ticker + " US Equity")
else:
    tickers = []  # Empty list if the file doesn't exist

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

data = []
current_year = current_date.year
previous_year = current_date - relativedelta(years=1)
previous_year_str = previous_year.strftime("%Y")

# Names for blpapi
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
FIELD_NAMES = [Name(field) for field in FIELDS_LIST]


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


def get_index_members(index, date=current_date):
    """
    Gets the current index members for the given index.
    """
    request = ref_data_service.createRequest("ReferenceDataRequest")
    request.append(SECURITIES, index)
    request.append(FIELDS, "INDX_MEMBERS")
    overrides = request.getElement(OVERRIDES)
    override1 = overrides.appendElement()
    override1.setElement(FIELDID, 'REFERENCE_DATE')
    override1.setElement(VALUE, date.strftime("%Y%m%d"))
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


def get_current_data(tickers, fields=FIELDS_LIST, date=current_date):
    """
    Gets current data for the given tickers.
    """

    data_rows = []
    for ticker in tqdm(tickers, desc='Fetching data'):
        request = ref_data_service.createRequest("ReferenceDataRequest")
        request.append(SECURITIES, ticker)
        for field in fields:
            request.append(FIELDS, field)
        overrides = request.getElement(OVERRIDES)
        override1 = overrides.appendElement()
        override1.setElement(FIELDID, 'REFERENCE_DATE')
        override1.setElement(VALUE, date.strftime("%Y%m%d"))

        session.sendRequest(request)

        event = event_loop(session)

        for msg in event:
            if msg.hasElement('securityData'):
                securityDataArray = msg.getElement('securityData')
                for securityData in securityDataArray.values():
                    if securityData.hasElement('fieldExceptions'):
                        field_exceptions = securityData.getElement(
                            'fieldExceptions')
                        if field_exceptions.numValues() > 0:
                            continue

                    field_data = securityData.getElement('fieldData')
                    data_row = {'Ticker': ticker}
                    for field in fields:
                        data_row[field] = fetch_field_data(field_data, field)

                    data_rows.append(data_row)

    return pd.DataFrame(data_rows)


to_get = tickers + get_index_members(INDEX)

# Get data for all tickers and fields
df = get_current_data(list(set(to_get)))

# Impute missing values by filling with mean
df_numeric = df.select_dtypes(include='number')
df_imputed = df_numeric.fillna(df_numeric.mean())
df[df_imputed.columns] = df_imputed

df.to_csv('current_data.csv', index=False)  # Save data to CSV file
