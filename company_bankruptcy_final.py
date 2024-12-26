# Company Bankruptcy Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Loading the data
data = pd.read_csv('american_bankruptcy.csv')

"""## Description of Columns

- X1	Current assets - All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year
- X2	Cost of goods sold - The total amount a company paid as a cost directly related to the sale of products
- X3	Depreciation and amortization - Depreciation refers to the loss of value of a tangible fixed asset over time (such as property, machinery, buildings, and plant). Amortization refers to the loss of value of intangible assets over time.
- X4	EBITDA - Earnings before interest, taxes, depreciation, and amortization. It is a measure of a company's overall financial performance, serving as an alternative to net income.
- X5	Inventory - The accounting of items and raw materials that a company either uses in production or sells.
- X6	Net Income - The overall profitability of a company after all expenses and costs have been deducted from total revenue.
- X7	Total Receivables - The balance of money due to a firm for goods or services delivered or used but not yet paid for by customers.
- X8	Market value - The price of an asset in a marketplace. In this dataset, it refers to the market capitalization since companies are publicly traded in the stock market.
- X9	Net sales - The sum of a company's gross sales minus its returns, allowances, and discounts.
- X10	Total assets - All the assets, or items of value, a business owns.
- X11	Total Long-term debt - A company's loans and other liabilities that will not become due within one year of the balance sheet date.
- X12	EBIT - Earnings before interest and taxes.
- X13	Gross Profit - The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services.
- X14	Total Current Liabilities - The sum of accounts payable, accrued liabilities, and taxes such as Bonds payable at the end of the year, salaries, and commissions remaining.
- X15	Retained Earnings - The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and its dividends to shareholders.
- X16	Total Revenue - The amount of income that a business has made from all sales before subtracting expenses. It may include interest and dividends from investments.
- X17	Total Liabilities - The combined debts and obligations that the company owes to outside parties.
- X18	Total Operating Expenses - The expenses a business incurs through its normal business operations.

## Exploratory Data Analysis (EDA)
"""

# Checking the total rows and columns
print(f'No. of Rows: {data.shape[0]} \nNo. of Columns: {data.shape[1]}')

data.company_name.unique() # Checking the number of unique companies

# Checking the datatype of each column and if there are any null values
data.info()

# Data Description
data.describe(include='all').fillna('-')

# Checking if there are any duplicate values
data.duplicated().sum()

# Converting the datatype of year column from float to datetime
df = pd.DataFrame({'year':data.year,'month': '1','day':'1'})

data1 = data.copy()
data1.year = pd.to_datetime(df)

data1.info()

# Filtering out the columns containing numerical values
for i in data1.columns:
    col = data1.select_dtypes('number')

col

## Feature Scaling

from sklearn.preprocessing import MinMaxScaler

# Feature scaling using Min Max Scaler
mms = MinMaxScaler()

data_new = pd.DataFrame(mms.fit_transform(data1.select_dtypes('number')), columns=col.columns)

# Filtering columns other than numerical values
for i in data1.columns:
    col1 = data1.select_dtypes(['object','datetime'])
col1

data2 = col1.join(data_new) # Creating the final dataset after feature scaling

# Checking for Correlation

plt.scatter(data2.X1, data2.X2)
plt.show()

data2.X1.corr(data2.X2)

corr_matrix = data2.select_dtypes('number').corr()
corr_matrix

sns.heatmap(corr_matrix)
plt.show()

# Checking for Outliers

plt.figure(figsize=(15, 20))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#ffa07a', '#32cd32', '#ff69b4', '#00ced1', '#9932cc']

num_colors = len(colors)

c = 1
for i, col2 in enumerate(data2.select_dtypes(float).columns):
    plt.subplot(10, 3, c)

    train_color = colors[c % num_colors]

    sns.boxplot(data=data1[col2], color=train_color)

    plt.xlabel('Dataset')
    plt.ylabel(col2)
    plt.title(f'Boxplot of {col2}')

    c += 1

plt.tight_layout()
plt.show()

# Function to detect outliers using IQR
def outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile
    Q3 = data[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers

# Treating the Outliers

# Function to treat outliers
def treat_outliers(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile
    Q3 = data[column].quantile(0.75)  # Third quartile
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

for col in data2.select_dtypes('number').columns:
    treat_outliers(data2, col)

plt.figure(figsize=(15, 20))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#ffa07a', '#32cd32', '#ff69b4', '#00ced1', '#9932cc']

num_colors = len(colors)

c = 1
for i, col2 in enumerate(data2.select_dtypes('number').columns):
    plt.subplot(10, 3, c)

    train_color = colors[c % num_colors]

    sns.boxplot(data=data2[col2], color=train_color)

    plt.xlabel('Dataset')
    plt.ylabel(col2)
    plt.title(f'Boxplot after outlier treatment of {col2}')

    c += 1

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=data1["X1"], color='skyblue')
plt.title('Before Outlier Treatment')
plt.subplot(1, 2, 2)
sns.boxplot(x=data2["X1"], color='lightgreen')
plt.title('After Outlier Treatment')
plt.show()

# Encoding the Object columns

b = pd.get_dummies(data2.company_name, dtype=int) # One Hot Encoding on Company Name column

data3 = data2.drop('company_name',axis=1)

# Importing Label Encoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

enc = pd.Series(le.fit_transform(data3.status_label), name='status') # Label Encoding on target variable

data_final = data3.join(enc).drop('status_label',axis=1) # Final data to be used to build model


data_final['year'] = data['year']

data_final.to_csv('data_final.csv') # Downloading the dataset

# Random Forest Model

# Importing necessary components
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

rf_classifier = RandomForestClassifier() # Random Forest algorithm

# Dividing the data into features & target variable
X, Y = data_final.drop(['year','status'], axis=1), data_final.status

X.head()

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)   #Splitting the dataset into training and testing

x_train

y_train

# Model fitting
model = rf_classifier.fit(x_train, y_train)

# Predict the target values for the test set
y_pred = rf_classifier.predict(x_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Cross validation

# K-Fold Cross Validation
kf_cv = KFold(4)

accuracy = {}

for i, (train_index,test_index) in enumerate(kf_cv.split(X)):
    train_x = X.loc[train_index]
    train_y = Y.loc[train_index]
    test_x = X.loc[test_index]
    test_y = Y.loc[test_index]
    rf_classifier.fit(train_x,train_y)
    pred = rf_classifier.predict(test_x)
    accuracy[f'split_no: {i}'] = accuracy_score(test_y,pred)

# Cross validation evaluation
from sklearn.model_selection import cross_val_score

cross_val = cross_val_score(rf_classifier,X,Y,cv=4)
np.mean(cross_val)

# Final Model Download

# Saving the model
import joblib

joblib.dump(model, 'rf_classifier.pkl')