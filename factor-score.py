import pandas as pd
import blpapi  # The Bloomberg API
import datetime

session = blpapi.Session()  # Start a Bloomberg session
session.start()

# Open the reference data service
if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata")
    session.stop()
    exit()

ref_data_service = session.getService("//blp/refdata")

fields = ['PX_LAST', 'CUR_MKT_CAP', 'BOOK_VAL_PER_SH', 'RETURN_COM_EQY', 'CF_FREE_CASH_FLOW']  # Bloomberg fields

data = []
current_year = datetime.datetime.now().year
index = "SPX Index"  # S&P 500 Index

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
                member_string = indx_member.getElementAsString('Member Ticker and Exchange Code')
                member_string = member_string.replace(" UW", " US Equity").replace(" UN", " US Equity")
                members.append(member_string)

    print(f"members: {members[:5]}...")
    return members

tickers = get_index_members(index, current_year)  # Get all S&P 500 members for the current year

for ticker in tickers:
    request = ref_data_service.createRequest("ReferenceDataRequest")
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
    # print(security_data_array)
    for i in range(security_data_array.numValues()):
        # Stick the data into data to be converted to a dataframe later
        security_data = security_data_array.getValueAsElement(i)
        field_data = security_data.getElement('fieldData')
        # print(field_data)
        market_cap = field_data.getElementAsFloat('CUR_MKT_CAP')
        book_value_per_share = field_data.getElementAsFloat('BOOK_VAL_PER_SH')
        roe = field_data.getElementAsFloat('RETURN_COM_EQY')
        free_cash_flow = field_data.getElementAsFloat('CF_FREE_CASH_FLOW')

        data.append([ticker, market_cap, book_value_per_share, roe, free_cash_flow])

df = pd.DataFrame(data, columns=['Ticker', 'MarketCap', 'BookValuePerShare', 'ROE', 'FreeCashFlow'])

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