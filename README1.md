# Predictive Analysis Using Social Profile in Online P2P Lending 
## 1. Introduction
Welcome to the Predictive Modelling Using Social Profile in Online P2P Lending Platform project! This repository contains the code and documentation for our study, which explores the determinants of performance predictability in the online peer-to-peer (P2P) lending market.

Online P2P lending platforms have revolutionized the financial landscape by enabling individual consumers to borrow and lend money directly to one another. Our research focuses on understanding the factors that influence the predictability of borrower rates and the likelihood of timely loan repayment and eligible borrower. 

The empirical analysis conducted in this study is based on a dataset comprising 9479 completed P2P transactions from the year 2007. Our findings demonstrate that combining financial data with social indicators significantly enhances the ability to predict performance in the P2P lending market. While social strength attributes do affect borrower rates and loan statuses, their impact is relatively minor compared to financial strength attributes.

This project aims to provide valuable insights and practical recommendations for both borrowers and lenders to increase their chances of successful funding and repayment in the P2P lending market. Additionally, we discuss potential future research opportunities to further eligible borrower field.



What we Do: 
|-------------------|
| Tasks: EDA (Exploratory Data Analysis):| 
| 1. Perform Data Exploration. 
| 2. Data Cleaning. 
| 3. Data Visualization and Manipulation. 
| 4. Formula for Target variable.
| 5. Feature Engineering
| 6. Trained the model using machine learning algorithm 
| 7. Hyperparameter Tuning: Optimize the hyperparameters to improve model performance.
| 8. Cross-Validation: Implement cross-validation techniques to ensure the model's robustness and generalizability.
| 9. Test and Evaluate the model
| 10. Building pipeline
| 11. Save and Serilize the model
| 12. Deployment of ML model
 
--
## 2. Data Preprocessing 

--
## 3. Data Visualization and Manipulation.
## Univariate Plots Section ##

A univariate plot depicts and summarizes the data's distribution. Individual observations are displayed on a dot plot, also known as a strip plot. 
![Univariate(1)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/e2865d96-0102-40d0-8a8c-d8a8054d1d2d)

## Bivariate Plots Section ##

Bivariate analysis is a type of statistical analysis in which two variables are compared to one another. One variable will be dependent, while the other will be independent. X and Y represent the variables. The differences between the two variables are analyzed to determine the extent to which the change has occurred.


![Bivariate_analysis](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/de399e95-c39d-4914-9e61-343fa0cec8e1)

![Bivariate_analysis(3)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/0ccc1807-721a-4682-8415-63fd711ac25a)
![Bivariate_analysis(4)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/3b62d064-da15-41dd-952d-a4900c273d5f)
![Bivariate_analysis(5)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/8a30d725-c2f3-4251-8d8e-092fb6535877)
![Bivariate_analysis(6)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/0789a8eb-e65e-4d02-95f2-3e6c90819c49)
![Bivariate_analysis(7)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/b73d05b8-2f91-4e04-8b4d-ffd9db841271)
![Bivariate_analysis(8)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/f5dd7d08-970a-409b-b274-b25133afbbeb)

## Multivariate Plots ##
Multivariate analysis is traditionally defined as the statistical study of experiments in which multiple measurements are taken on each experimental unit and the relationship between multivariate measurements and their structure is critical for understanding the experiment.
So, let's look at an example of a research question that we solved using multivariate plots and matplotlib.


## 4. Formula for Target variable
#### Loan Tenure in Months=(Year of ClosedDate−Year of LoanOriginationDate)×12+(Month of ClosedDate−Month of LoanOriginationDate)
```data['Loan_tenure'] = (data['ClosedDate'].dt.year - data['LoanOriginationDate'].dt.year) * 12 + (data['ClosedDate'].dt.month - data['LoanOriginationDate'].dt.month)```



### For Multi regression model we need three target Variables as one of them is the EMI.Below is the formula for creating the EMI column:

#### EMI= P * r * (1+r)^n / (1+r)^n -1
#### Where:
#### 𝑃 P is the principal amount (loan amount)
#### 𝑟 r is the monthly interest rate.
#### 𝑛 n is the tenure in months.
 

```def cal_EMI(principal, annual_rate, tenure_months):
    # Check if tenure_months is zero
    if tenure_months == 0:
        return 0
    # Calculate the monthly interest rate
    monthly_rate = annual_rate / (12 * 100)
    # Check if monthly_rate is zero to avoid division by zero
    if monthly_rate == 0:
        return principal / tenure_months
    # Calculate the EMI using the formula
    try:
        emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / (((1 + monthly_rate) ** tenure_months) - 1)
    except ZeroDivisionError:
        emi = 0  # In case of any unexpected zero division, return 0 as EMI
    return emi
```

#### Creating the eligible Loan amount column as this also the target variable:
#### Totalpaymentdue:
#### Total Payment Due=(P+(P× R/100 ​ ))×T
#### Max Allowable Amount=M×12×0.30
#### Eligible Loan Amount= { p if Total Payment Due≤Max Allowable Amount }

``` def calculate_eligible_loan_amount(row):
    total_payment_due = calculate_total_payment_due(row["LP_CustomerPrincipalPayments"], row["BorrowerRate"], row["Loan_tenure"])
    max_allowable_amount = calculate_max_allowable_amount(row["StatedMonthlyIncome"])

    if total_payment_due <= max_allowable_amount:
        return row["LP_CustomerPrincipalPayments"]
    else:
        return 0 
```
--

### Creating another important variable PROI(Preferred rate of intrest)
#### ROI (Return on Investment):
#### ROI = InterestAmount TotalAmount ROI= InterestAmount/TotalAmount
#### PROI=Median(ROI) ​
``` # Calculate ROI
    data['InterestAmount'] = data['LoanOriginalAmount'] * data['BorrowerRate']
    data['TotalAmount'] = data['InterestAmount'] + data['LoanOriginalAmount']
    data['ROI'] = data['InterestAmount'] / data['TotalAmount']
    print(data['ROI'].describe())
```

--

5. Feature Engineering


#### Extracting the features.
###### we have to extract the date,month,year from the datetime columns

### Function to select top 10 features based on MI scores for each target
Top 10 features for EMI:                   Top 10 features for EligibleLoanAmount:            Top 10 features for PROI:
|TotalPaymentDue                             |LP_CustomerPrincipalPayments                     LP_CustomerPrincipalPayments
|LP_CustomerPrincipalPayments                |TotalPaymentDue                                  LP_GrossPrincipalLoss
|LoanOriginalAmount                          |LP_CustomerPayments                              LP_NetPrincipalLoss
|Loan_tenure                                 |LP_GrossPrincipalLoss                            LoanFirstDefaultedCycleNumber
|LP_CustomerPayments                         |LP_NetPrincipalLoss                              LoanCurrentDaysDelinquent
|TotalAmount                                 |LP_ServiceFees                                   TotalPaymentDue
|LP_ServiceFees                              |Loan_tenure                                      MonthlyLoanPayment
|MonthlyLoanPayment                          |LoanOriginalAmount                               LP_CustomerPayments 
|InterestAmount                              |TotalAmount                                      TotalAmount
|LP_GrossPrincipalLoss                       |LoanFirstDefaultedCycleNumber                    LoanOriginalAmount
|LP_NetPrincipalLoss                         |LP_InterestandFees                               InterestAmount
|ROI                                         |MonthlyLoanPayment                               Loan_tenure
|BorrowerRate                                |LoanCurrentDaysDelinquent                        LP_ServiceFees
|LenderYield                                 |InterestAmount                                   BorrowerAPR
|BorrowerAPR                                 |BorrowerAPR                                      ROI
|LP_InterestandFees                          |ROI                                              LP_InterestandFees
|LoanFirstDefaultedCycleNumber               |LenderYield                                      LenderYield
|EstimatedLoss                               |BorrowerRate                                     BorrowerRate
|EstimatedEffectiveYield                     |Closeddate_year                                  EstimatedEffectiveYield
|rosperRating (numeric)                      |EstimatedEffectiveYield                          EstimatedReturn

--

# 6. Trained the model using machine learning algorithm 

### XGBoost classifier
# Standardize the features
``` scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
n_components = 30  # Number of principal components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train XGBoost classifier
xgb_clf = XGBClassifier(random_state=42)
xgb_clf.fit(X_train_pca, y_train) ```

``` Accuracy: 0.9633176898764486
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.94      0.93      2893
           1       0.98      0.97      0.97      7548

    accuracy                           0.96     10441
   macro avg       0.95      0.96      0.95     10441
weighted avg       0.96      0.96      0.96     10441
```
--
### RandomForestRegressor
This evaluation demonstrates the effective use of a multi-output regression model to predict multiple target variables using a Random Forest Regressor. The performance metrics provide insights into the model's predictive accuracy and reliability. Further analysis and tuning can be done to enhance the model's performance.
Data Preprocessing
    Data Splitting: The dataset is divided into training and testing sets using an 80-20 split.
    Scaling: The features are standardized using StandardScaler to ensure consistent scaling across the data.
Model Training
A RandomForestRegressor is used as the base estimator within a MultiOutputRegressor. The model is trained on the scaled training data.    
Performance Evaluation
The performance of the model is evaluated using Mean Squared Error (MSE) and R² score for each target variable individually, as well as overall metrics.
Individual Target Variable Performance
For each target variable, the following metrics are calculated:
    Mean Squared Error (MSE)
    R² Score

***Splitting Data***
The dataset is divided into features and target variables:
    Features (X): All columns except EMI, EligibleLoanAmount, PROI, and LoanStatus.
    Targets:
        Binary classification target (y_class): LoanStatus
        Multi-output regression targets (y_multi): EMI, EligibleLoanAmount, PROI

***Model Training***
Binary Classification
Model: XGBoost Classifier with SMOTE and PCA.
Pipeline steps:
    Preprocessor: Preprocesses the data
    SMOTE: Balances the dataset
    PCA: Reduces dimensionality to 30 components
    Classifier: XGBoost classifier

    ```classification_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=30, random_state=42)),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])
    ```

_Multi-Output Regression_
Model: MultiOutputRegressor with RandomForestRegressor.
Pipeline steps:
    Preprocessor: Preprocesses the augmented data
    Regressor: MultiOutputRegressor with RandomForestRegressor
    ```regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_augmented),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
    ])
     ```


_Combined Pipeline_
The CombinedPipeline class integrates both pipelines, where the output of the classification pipeline augments the input data for the regression pipeline.

Methods
    fit: Trains both the classification and regression models.
    predict: Makes predictions for the regression targets.
    evaluate: Evaluates the regression model performance using MSE and R² score.

    Training
    The combined pipeline is trained using:
    ```combined_pipeline.fit(X_train, y_train_class, y_train_multi)```

    Evaluation
    The model's performance is evaluated by predicting and calculating metrics:
    ```combined_pipeline.evaluate(X_test, y_test_multi)```


## 9.  Deployment
### 9.1.1 Streamlit

The development process of the web application was Developed into the Streamlit.
Streamlit is an open source app framework in Python language. 
It helps us create web apps for data science and machine learning in a short time.
It is compatible with major Python libraries such as scikit-learn, NumPy, pandas, Matplotlib etc. 
