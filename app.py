import streamlit as st
# from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle 
import joblib
from streamlit_option_menu import option_menu

model = pickle.load(open('model_file.sav','rb'))
# model = pk.load(open('E://p2/combined_pipeline.pkl', 'rb'))
# scaler =pk.load(open('C:/Users/DELL/OneDrive/Documents/p2p/scaler.pkl', 'rb'))
   
# def main():
st.header('Loan Prediction App')
html_temp = """
<div style="background-color: blue; padding: 10px;">
<h2 style="color: white; text-align: center;">Prediction of Loan Eligibility</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
    
    # User input fields
MonthlyLoanPayment = st.text_input("Monthly Loan Payment", placeholder="Enter EMI")
Loan_tenure = st.text_input("Loan Tenure", placeholder="Loan Tenure")
ROI = st.text_input("Rate of Interest", placeholder="Enter ROI")
BorrowerAPR = st.text_input("Borrower APR", placeholder="Enter Borrower APR")
BorrowerRate = st.text_input("Borrower Rate", placeholder="Enter Borrower Rate")
LP_CustomerPayments = st.text_input("LP Customer Payments", placeholder="Enter LP Customer Payments")
LP_CustomerPrincipalPayments = st.text_input("LP Customer Principal Payments", placeholder="Enter LP Customer Principal Payments")
LP_GrossPrincipalLoss = st.text_input("LP Gross Principal Loss", placeholder="Enter LP Gross Principal Loss")
LP_NetPrincipalLoss = st.text_input("LP Net Principal Loss", placeholder="Enter LP Net Principal Loss")
LoanFirstDefaultedCycleNumber = st.text_input("Loan First Defaulted Cycle Number", placeholder="Enter Loan First Defaulted Cycle Number")
LoanOriginalAmount = st.text_input("Loan Original Amount", placeholder="Enter Loan Original Amount")
LoanCurrentDaysDelinquent = st.text_input("Loan Current Days Delinquent", placeholder="Enter Loan Current Days Delinquent")
TotalPaymentDue = st.text_input("Total Payment Due", placeholder="Enter Total Payment Due")
InterestAmount = st.text_input("Interest Amount", placeholder="Enter Interest Amount")
TotalAmount = st.text_input("Total Amount", placeholder="Enter Total Amount")

    # Prediction button
if st.button("Predict"):
        if model is not None and scaler is not None:
            try:
                # Convert inputs to float and validate
                inputs = np.array([
                    float(MonthlyLoanPayment or 0),
                    float(Loan_tenure or 0),
                    float(ROI or 0),
                    float(BorrowerAPR or 0),
                    float(BorrowerRate or 0),
                    float(LP_CustomerPayments or 0),
                    float(LP_CustomerPrincipalPayments or 0),
                    float(LP_GrossPrincipalLoss or 0),
                    float(LP_NetPrincipalLoss or 0),
                    float(LoanFirstDefaultedCycleNumber or 0),
                    float(LoanOriginalAmount or 0),
                    float(LoanCurrentDaysDelinquent or 0),
                    float(TotalPaymentDue or 0),
                    float(InterestAmount or 0),
                    float(TotalAmount or 0)
                ]).reshape(1, -1)

                # Debugging outputs
                st.write("Input shape:", inputs.shape)
                st.write("Input data:", inputs)

                # Scaling inputs (assuming scaler is defined and loaded)
                inputs_scaled = scaler.transform(inputs)

                # Predict using the model
                predictions = model.predict(inputs_scaled)
                PEMI, PROI, ELA = predictions[0]

                st.success(f"Predicted PEMI: {PEMI}")
                st.success(f"Predicted PROI: {PROI}")
                st.success(f"Eligible Loan Amount: {ELA}")

            except ValueError as e:
                st.error(f"Value error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Model or scaler not loaded. Cannot make predictions.")

    # Heading for user input result
st.header("User Input Result")

    # Display the entered loan amount if not empty
if MonthlyLoanPayment:
        st.write("Monthly Loan Payment:", MonthlyLoanPayment)

    # Display the entered loan tenure if not empty
if Loan_tenure:
        st.write("Loan Tenure:", Loan_tenure)

    # Display the entered ROI if not empty
if ROI:
        st.write("Rate of Interest:", ROI)

    # Display the entered Borrower APR if not empty
if BorrowerAPR:
        st.write("Borrower APR:", BorrowerAPR)

    # Display the entered Borrower Rate if not empty
if BorrowerRate:
        st.write("Borrower Rate:", BorrowerRate)

    # Display the entered LP Customer Payments if not empty
if LP_CustomerPayments:
        st.write("LP Customer Payments:", LP_CustomerPayments)

    # Display the entered LP Customer Principal Payments if not empty
if LP_CustomerPrincipalPayments:
        st.write("LP Customer Principal Payments:", LP_CustomerPrincipalPayments)

    # Display the entered LP Gross Principal Loss if not empty
if LP_GrossPrincipalLoss:
        st.write("LP Gross Principal Loss:", LP_GrossPrincipalLoss)

    # Display the entered LP Net Principal Loss if not empty
if LP_NetPrincipalLoss:
        st.write("LP Net Principal Loss:", LP_NetPrincipalLoss)

    # Display the entered Loan First Defaulted Cycle Number if not empty
if LoanFirstDefaultedCycleNumber:
        st.write("Loan First Defaulted Cycle Number:", LoanFirstDefaultedCycleNumber)

    # Display the entered Loan Original Amount if not empty
if LoanOriginalAmount:
        st.write("Loan Original Amount:", LoanOriginalAmount)

    # Display the entered Loan Current Days Delinquent if not empty
if LoanCurrentDaysDelinquent:
        st.write("Loan Current Days Delinquent:", LoanCurrentDaysDelinquent)

    # Display the entered Total Payment Due if not empty
if TotalPaymentDue:
        st.write("Total Payment Due:", TotalPaymentDue)

    # Display the entered Interest Amount if not empty
if InterestAmount:
        st.write("Interest Amount:", InterestAmount)

    # Display the entered Total Amount if not empty
if TotalAmount:
        st.write("Total Amount:", TotalAmount)

# if __name__ == '__main__':
#     main()
