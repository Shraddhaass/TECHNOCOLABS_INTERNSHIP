import streamlit as st
import pickle
import os
from pipeline import CombinedPipeline
import numpy as np
import pandas as pd



model = pickle.load(open('combined_pipeline.joblib','rb'))
scaler= pickle.load(open('scaler.pkl','rb'))

# Define the expected columns used during training
expected_columns = [
    'CreditGrade', 'Term', 'BorrowerAPR', 'BorrowerRate', 'LenderYield',
    'EstimatedEffectiveYield', 'EstimatedLoss', 'EstimatedReturn',
    'ProsperRating (numeric)', 'ProsperScore', 'ListingCategory (numeric)',
    'BorrowerState', 'Occupation', 'EmploymentStatus',
    'EmploymentStatusDuration', 'IsBorrowerHomeowner', 'CurrentlyInGroup',
    'CreditScoreRangeLower', 'CreditScoreRangeUpper', 'CurrentCreditLines',
    'OpenCreditLines', 'TotalCreditLinespast7years',
    'OpenRevolvingAccounts', 'OpenRevolvingMonthlyPayment',
    'InquiriesLast6Months', 'TotalInquiries', 'CurrentDelinquencies',
    'AmountDelinquent', 'DelinquenciesLast7Years',
    'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
    'RevolvingCreditBalance', 'BankcardUtilization',
    'AvailableBankcardCredit', 'TotalTrades',
    'TradesNeverDelinquent (percentage)', 'TradesOpenedLast6Months',
    'DebtToIncomeRatio', 'IncomeRange', 'IncomeVerifiable',
    'StatedMonthlyIncome', 'TotalProsperLoans',
    'TotalProsperPaymentsBilled', 'OnTimeProsperPayments',
    'ProsperPaymentsLessThanOneMonthLate',
    'ProsperPaymentsOneMonthPlusLate', 'ProsperPrincipalBorrowed',
    'ProsperPrincipalOutstanding', 'ScorexChangeAtTimeOfListing',
    'LoanCurrentDaysDelinquent', 'LoanFirstDefaultedCycleNumber',
    'LoanMonthsSinceOrigination', 'LoanNumber', 'LoanOriginalAmount',
    'MonthlyLoanPayment', 'LP_CustomerPayments',
    'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees',
    'LP_CollectionFees', 'LP_GrossPrincipalLoss', 'LP_NetPrincipalLoss',
    'LP_NonPrincipalRecoverypayments', 'PercentFunded', 'Recommendations',
    'InvestmentFromFriendsCount', 'InvestmentFromFriendsAmount',
    'Investors', 'Loan_tenure', 'TotalPaymentDue', 'MaxAllowableAmount',
    'InterestAmount', 'TotalAmount', 'ROI', 'ListingCreationDate_Year',
    'ListingCreationDate_Month', 'ListingCreationDate_Day',
    'DateCreditPulled_Year', 'DateCreditPulled_Month',
    'DateCreditPulled_Day', 'LoanOriginationDate_Year',
    'LoanOriginationDate_Month', 'LoanOriginationDate_Day',
    'Closeddate_year', 'Closeddate_Month', 'Closeddate_Day'
]

# Function to prepare input data as a Pandas DataFrame
def prepare_own_data(input_data):
    input_features = {
        'LP_CustomerPrincipalPayments': input_data.get('LP_CustomerPrincipalPayments', 0),
        'TotalPaymentDue': input_data.get('TotalPaymentDue', 0),
        'LP_CustomerPayments': input_data.get('LP_CustomerPayments', 0),
        'LP_GrossPrincipalLoss': input_data.get('LP_GrossPrincipalLoss', 0),
        'LP_NetPrincipalLoss': input_data.get('LP_NetPrincipalLoss', 0),
        'LoanFirstDefaultedCycleNumber': input_data.get('LoanFirstDefaultedCycleNumber', 0),
        'MonthlyLoanPayment': input_data.get('MonthlyLoanPayment', 0),
        'TotalAmount': input_data.get('TotalAmount', 0),
        'LoanOriginalAmount': input_data.get('LoanOriginalAmount', 0),
        'InterestAmount': input_data.get('InterestAmount', 0),
        'Loan_tenure': input_data.get('Loan_tenure', 0),
        'LP_ServiceFees': input_data.get('LP_ServiceFees', 0),
        'BorrowerAPR': input_data.get('BorrowerAPR', 0),
        'ROI': input_data.get('ROI', 0),
        'BorrowerRate': input_data.get('BorrowerRate', 0),
        # 'Closeddate_Day': input_data.get('Closeddate_Day', 0),
        # 'Closeddate_Month': input_data.get('Closeddate_Month', 0),
        # 'Closeddate_year': input_data.get('Closeddate_year', 0)
    }

    # Assuming expected_columns includes these features
    user_df = pd.DataFrame([input_features], columns=expected_columns)

    return user_df







st.header('Loan Prediction App')
html_temp = """
<div style="background-color: blue; padding: 10px;">
<h2 style="color: white; text-align: center;">Prediction of Loan Eligibility</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
 # User input fields
LP_CustomerPrincipalPayments = st.text_input("LP Customer Principal Payments", placeholder="Enter LP Customer Principal Payments")
TotalPaymentDue = st.text_input("Total Payment Due", placeholder="Enter Total Payment Due")
LP_CustomerPayments = st.text_input("LP Customer Payments", placeholder="Enter LP Customer Payments")
LP_GrossPrincipalLoss = st.text_input("LP Gross Principal Loss", placeholder="Enter LP Gross Principal Loss")
LP_NetPrincipalLoss = st.text_input("LP Net Principal Loss", placeholder="Enter LP Net Principal Loss")
LoanFirstDefaultedCycleNumber = st.text_input("Loan First Defaulted Cycle Number", placeholder="Enter Loan First Defaulted Cycle Number")
MonthlyLoanPayment = st.text_input("Monthly Loan Payment", placeholder="Enter EMI")
TotalAmount = st.text_input("Total Amount", placeholder="Enter Total Amount")
LoanOriginalAmount = st.text_input("Loan Original Amount", placeholder="Enter Loan Original Amount")
InterestAmount = st.text_input("Interest ", placeholder="Enter Interest Amount")
Loan_tenure = st.text_input("Loan Tenure", placeholder="Loan Tenure")
LP_ServiceFees = st.text_input("LP_ServiceFees", placeholder="Enter LP_ServiceFees")
BorrowerAPR = st.text_input("Borrower APR", placeholder="Enter Borrower APR")
ROI = st.text_input("Rate of Interest", placeholder="Enter ROI")
BorrowerRate = st.text_input("Borrower Rate", placeholder="Enter Borrower Rate")
# #Closeddate_Day = st.text_input("Closeddate_Day", placeholder="Enter Closeddate_Day")
# Closeddate_Month=st.text_input("Closeddate_Month", placeholder="Enter Closeddate_Month")
# Closeddate_year=st.text_input("Closeddate_year", placeholder="Enter Closeddate_year")


# Prediction button
if st.button("Predict"):
    if model is not None and scaler is not None:
        try:
            # Collect user inputs
            input_data = {
                'LP_CustomerPrincipalPayments': float(LP_CustomerPrincipalPayments or 0),
                'TotalPaymentDue': float(TotalPaymentDue or 0),
                'LP_CustomerPayments': float(LP_CustomerPayments or 0),
                'LP_GrossPrincipalLoss': float(LP_GrossPrincipalLoss or 0),
                'LP_NetPrincipalLoss': float(LP_NetPrincipalLoss or 0),
                'LoanFirstDefaultedCycleNumber': float(LoanFirstDefaultedCycleNumber or 0),
                'MonthlyLoanPayment': float(MonthlyLoanPayment or 0),
                'TotalAmount': float(TotalAmount or 0),
                'LoanOriginalAmount': float(LoanOriginalAmount or 0),
                'InterestAmount': float(InterestAmount or 0),
                'Loan_tenure': float(Loan_tenure or 0),
                'LP_ServiceFees': float(LP_ServiceFees or 0),
                'BorrowerAPR': float(BorrowerAPR or 0),
                'ROI': float(ROI or 0),
                'BorrowerRate': float(BorrowerRate or 0),
                'Closeddate_Day': float(Closeddate_Day or 0),  # Ensure these are included if used
                'Closeddate_Month': float(Closeddate_Month or 0),
                'Closeddate_year': float(Closeddate_year or 0)
            }

            # Prepare input data with 83 columns expected by the model
            user_df = prepare_own_data(input_data)

            # Scaling inputs
            inputs_scaled = scaler.transform(user_df)

            # Predict using the model
            predictions = model.predict(inputs_scaled)
            PEMI, PROI, ELA = predictions[0]

            st.markdown(f"""
            ### Predicted Values:
            - **Predicted EMI:** {PEMI}
            - **Predicted PROI:** {PROI}
            - **Predicted Eligible Loan Amount:** {ELA}
            """)

        except ValueError as e:
            st.error(f"Value error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Model or scaler not loaded. Cannot make predictions.")

























# if st.button("Predict"):
#     if model is not None and scaler is not None:
#         try:
#             # Convert inputs to float and validate
#             inputs = np.array([
#                 float(LP_CustomerPrincipalPayments or 0),
#                 float(TotalPaymentDue or 0),
#                 float(LP_CustomerPayments or 0),
#                 float(LP_GrossPrincipalLoss or 0),
#                 float(LP_NetPrincipalLoss or 0),
#                 float(LoanFirstDefaultedCycleNumber or 0),
#                 float(MonthlyLoanPayment or 0),
#                 float(TotalAmount or 0),
#                 float(LoanOriginalAmount or 0),
#                 float(InterestAmount or 0),
#                 float(Loan_tenure or 0),
#                 float(LP_ServiceFees or 0),
#                 float(BorrowerAPR or 0),
#                 float(ROI or 0),
#                 float(BorrowerRate or 0)
#             ]).reshape(1, -1)

#             # Debugging outputs
#             st.write("Input shape:", inputs.shape)
#             st.write("Input data:", inputs)

#             # Scaling inputs
#             inputs_scaled = scaler.transform(inputs)

#             # Predict using the model
#             predictions = model.predict(inputs_scaled)
#             PEMI, PROI, ELA = predictions[0]

#             st.markdown(f"""
#             ### Predicted Values:
#             - **Predicted EMI:** {PEMI}
#             - **Predicted PROI:** {PROI}
#             - **Predicted Eligible Loan Amount:** {ELA}
#             """)

#         except ValueError as e:
#             st.error(f"Value error: {e}")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#     else:
#         st.error("Model or scaler not loaded. Cannot make predictions.")






# Prediction button
# if st.button("Predict"):
#         if model is not None and scaler is not None:
#             try:
#                 # Convert inputs to float and validate
#                 inputs = np.array([
#                     float(LP_CustomerPrincipalPayments or 0),
#                     float(TotalPaymentDue or 0),
#                     float(LP_CustomerPayments or 0),
#                     float(LP_GrossPrincipalLoss or 0),
#                     float(LP_NetPrincipalLoss or 0),
#                     float(LoanFirstDefaultedCycleNumber or 0),
#                     float(MonthlyLoanPayment or 0),
#                     float(TotalAmount or 0),
#                     float(LoanOriginalAmount or 0),
#                     float(Loan_tenure or 0),
#                     float(LP_ServiceFees or 0),
#                     float(BorrowerAPR or 0),
#                     float(ROI or 0),
#                     float(BorrowerRate or 0)
#                 ]).reshape(1, -1)

#                 # Debugging outputs
#                 st.write("Input shape:", inputs.shape)
#                 st.write("Input data:", inputs)

#                 # Scaling inputs (assuming scaler is defined and loaded)
#                 inputs_scaled = scaler.transform(inputs)

#                 # Predict using the model
#                 predictions = model.predict(inputs_scaled)
#                 PEMI, PROI, ELA = predictions[0]

#                 st.success(f"Predicted PEMI: {PEMI}")
#                 st.success(f"Predicted PROI: {PROI}")
#                 st.success(f"Eligible Loan Amount: {ELA}")

#             except ValueError as e:
#                 st.error(f"Value error: {e}")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
#         else:
#             st.error("Model or scaler not loaded. Cannot make predictions.")