import blpapi
import numpy as np
import pandas as pd
import pickle
import argparse
import datetime

DEFAULT_START_DATE = "20200101"
DEFAULT_END_DATE = "20201231"

parser = argparse.ArgumentParser()
parser.add_argument("--date", nargs=2, type=str)
args = parser.parse_args()

if args.date:
    start_date = args.date[0]
    end_date = args.date[1]
    try:
        datetime.datetime.strptime(start_date, '%Y%m%d')
        datetime.datetime.strptime(end_date, '%Y%m%d')
    except ValueError:
        print("Invalid date format. Using default dates.")
        start_date = DEFAULT_START_DATE
        end_date = DEFAULT_END_DATE
else:
    start_date = DEFAULT_START_DATE
    end_date = DEFAULT_END_DATE

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Connect to the Bloomberg API
sessionOptions = blpapi.SessionOptions()
session = blpapi.Session(sessionOptions)
if not session.start():
    print("Failed to start session.")
    exit()
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    exit()
ref_data_service = session.getService("//blp/refdata")

# Define the S&P 500 index ticker
INDEX_TICKER = "SPX Index"

# Define constant names
SECURITIES = blpapi.Name("securities")
FIELDS = blpapi.Name("fields")
START_DATE = blpapi.Name("startDate")
END_DATE = blpapi.Name("endDate")
PERIOD_ADJUSTMENT = blpapi.Name("periodicityAdjustment")
PERIOD_SELECTION = blpapi.Name("periodicitySelection")
MAX_DATA_POINTS = blpapi.Name("maxDataPoints")
OVERRIDES = blpapi.Name("overrides")
SECURITY_DATA = blpapi.Name("securityData")
SECURITY = blpapi.Name("security")
FIELD_DATA = blpapi.Name("fieldData")
FIELDID = blpapi.Name("fieldId")
VALUE = blpapi.Name("value")

def event_loop(request):
    # Send the request and wait for the response
    session.sendRequest(request)
    while True:
        event = session.nextEvent()
        if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
            return event

def get_index_members(index, year):
    """
    Gets the index members for the given index and year.
    """
    print(f"Getting index members for {index} in {year}...")
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

# Get the normalized features for each year
def get_norm_features(data):
    #if data is not a dataframe or is empty, throw an exception
    if not isinstance(data, pd.DataFrame):
        raise Exception("Data must be a dataframe")
    if data.empty:
        raise Exception("Dataframe cannot be empty")
    #TODO: finish this function
    