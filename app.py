import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("rf_classifier.pkl")

# Load the dataset for feature reference
data = pd.read_csv("data_final.csv")
feature_names = list(data.columns[2:-1])  # First Column is index, Second Column is Year & Last column is the target

# Streamlit app title
st.title("Company Bankruptcy Prediction")

st.markdown("""
## Description of Columns

- X1:	Current assets - All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year
- X2:	Cost of goods sold - The total amount a company paid as a cost directly related to the sale of products
- X3:	Depreciation and amortization - Depreciation refers to the loss of value of a tangible fixed asset over time (such as property, machinery, buildings, and plant). Amortization refers to the loss of value of intangible assets over time.
- X4:	EBITDA - Earnings before interest, taxes, depreciation, and amortization. It is a measure of a company's overall financial performance, serving as an alternative to net income.
- X5:	Inventory - The accounting of items and raw materials that a company either uses in production or sells.
- X6:	Net Income - The overall profitability of a company after all expenses and costs have been deducted from total revenue.
- X7:	Total Receivables - The balance of money due to a firm for goods or services delivered or used but not yet paid for by customers.
- X8:	Market value - The price of an asset in a marketplace. In this dataset, it refers to the market capitalization since companies are publicly traded in the stock market.
- X9:	Net sales - The sum of a company's gross sales minus its returns, allowances, and discounts.
- X10:	Total assets - All the assets, or items of value, a business owns.
- X11:	Total Long-term debt - A company's loans and other liabilities that will not become due within one year of the balance sheet date.
- X12:	EBIT - Earnings before interest and taxes.
- X13:	Gross Profit - The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services.
- X14:	Total Current Liabilities - The sum of accounts payable, accrued liabilities, and taxes such as Bonds payable at the end of the year, salaries, and commissions remaining.
- X15:	Retained Earnings - The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and its dividends to shareholders.
- X16:	Total Revenue - The amount of income that a business has made from all sales before subtracting expenses. It may include interest and dividends from investments.
- X17:	Total Liabilities - The combined debts and obligations that the company owes to outside parties.
- X18:	Total Operating Expenses - The expenses a business incurs through its normal business operations.
""")
st.markdown("""
### Enter Company Financial Data
Provide input values for the features below to predict whether the company is at risk of bankruptcy.
""")

# Create input fields for each feature
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert inputs to DataFrame for prediction
input_data = pd.DataFrame([inputs])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("The company is at risk of bankruptcy.")
    elif prediction[0] == 0:
        st.success("The company is financially stable.")

