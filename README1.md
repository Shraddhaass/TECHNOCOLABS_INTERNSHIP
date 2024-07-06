# Predictive Analysis Using Social Profile in Online P2P Lending 
## 1. Introduction
Welcome to the Predictive Modelling Using Social Profile in Online P2P Lending Platform project! This repository contains the code and documentation for our study, which explores the determinants of performance predictability in the online peer-to-peer (P2P) lending market.

Online P2P lending platforms have revolutionized the financial landscape by enabling individual consumers to borrow and lend money directly to one another. Our research focuses on understanding the factors that influence the predictability of borrower rates and the likelihood of timely loan repayment and eligible borrower. 

The empirical analysis conducted in this study is based on a dataset comprising 9479 completed P2P transactions from the year 2007. Our findings demonstrate that combining financial data with social indicators significantly enhances the ability to predict performance in the P2P lending market. While social strength attributes do affect borrower rates and loan statuses, their impact is relatively minor compared to financial strength attributes.

This project aims to provide valuable insights and practical recommendations for both borrowers and lenders to increase their chances of successful funding and repayment in the P2P lending market. Additionally, we discuss potential future research opportunities to further eligible borrower field.

## What We Do

# Loan Prediction Model

## What we done:
| Tasks                                        |
|----------------------------------------------|
| 1. Perform Data Exploration                  |
| 2. Visualization graphs                      |
|    - EDA (Exploratory Data Analysis):        |
|      1. Univariate Analysis                  |
|      2. BiVariate Analysis                   |
|      3. Multivariate Analysis                |
| 3. Feature Engineering                       |
| 4. Model Evaluation                          |
|    - (a) AdaBoost Classifier                 |
|    - (b) SVM Classifier                      |
|    - (c) XGBoost Classifier                  |
|    - (d) Decision Tree Classifier            |
| 5. Formula for Target Variable               |
| 6. Train the Model Using Machine Learning Algorithms |
|    - Random Forest Classifier                |
|    - MultiOutputRegressor                    |
| 7. Test and Evaluate the Model               |
| 8. Build Pipeline                            |
| 9. Save and Serialize the Model              |
| 10. Deployment of ML Model                   |

--
## 2. Data Preprocessing 
       Exploratory Data Analysis
                 1.Clear Null Values
                 2.Encoding Categorical Data
                 3.Clearing Outliers
                 4.Formating columns having Datetime
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
![Multivariate_analysis(1)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/40e4afd8-c93c-4ea0-a576-e5cb3f4462d1)
![Multivariate_analysis(2)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/35b31e3c-5d61-4b0f-9939-ac9a5cbde306)
![Multivariate_analysis(3)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/18f338c2-01bb-44f4-8dd3-a349f228d91d)
![Multivariate_analysis(4)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/f456c3f0-bc90-40b5-bd58-8c4d9ff7b4bc)
![Multivariate_analysis(5)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/f1a6285b-cfde-4d07-b646-e8e6ee2c9152)
![Multivariate_analysis(6)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/bc3b23dc-e8f8-41df-a45b-448cd7a302d8)
![Multivariate_analysis(6)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/0467a449-c799-4a7c-8806-f8aeb2332a0b)
![Multivariate_analysis(7)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/b40e49a7-7813-4255-b7be-d0e1b9d47037)
![Multivariate_analysis(8)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/b3cb1ab2-78d1-49ff-b2e5-40accf924de6)
![Multivariate_analysis(9)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/5e1143ad-7b92-49f5-92ad-40fa3ea234ac)
![Multivariate_analysis(10)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/d390db2d-ae66-471c-854b-dfc196148a07)
![Multivariate_analysis(11)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/0c961dd1-4439-4df3-8f77-06ee3fb59f31)
![Multivariate_analysis(12)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/8d4769b3-5d9b-4fe0-a7e3-a889e3b052fe)
![Multivariate_analysis(13)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/90b35661-eb70-4edb-a9cb-f2c3515e767d)
![Multivariate_analysis(14)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/312fa96e-2db7-4d2f-8897-76433c3e8b05)
![Multivariate_analysis(15)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/64aa8587-b666-491c-b72e-7d558c8c2173)
![Multivariate_analysis(16)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/40acd8fb-1759-4857-aca1-4dd40aee362c)
![Multivariate_analysis(17)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/6bb2b0eb-4426-4f92-806d-f0c621c70edc)
![Multivariate_analysis(18)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/92e19d6c-4c0b-40b5-9178-a1a6d19572b7)
![Multivariate_analysis(19)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/d0448702-8fd8-4d21-baf6-2205fdec2be2)

# Model we work
## Model Evaluation

### 1. XGBoost Classifier
- **Accuracy:** 95.23%
- **Confusion Matrix:** [[2563, 343], [722, 18719]]
- **Precision:** Class 0: 78%, Class 1: 98%
- **Recall:** Class 0: 88%, Class 1: 96%
- **F1 Score:** Class 0: 83%, Class 1: 97%
- **Support:** Class 0: 2906, Class 1: 19441
- **Summary:** High precision, recall, and F1-score for both classes, indicating effective classification of Default and Non-Default loans.

### 2. SVM Classifier
- **Accuracy:** 95.77%
- **Confusion Matrix:** [[2342, 564], [381, 19060]]
- **Precision:** 97.13%
- **Recall:** 98.04%
- **F1 Score:** 97.58%
- **Summary:** Strong performance in terms of accuracy, precision, recall, and F1 score, making it a viable option for classification tasks.

### 3. AdaBoost Classifier
- **Accuracy:** 93.44%
- **Confusion Matrix:** [[2582, 324], [1143, 18298]]
- **Precision:** 94.50%
- **Recall:** 93.44%
- **F1 Score:** 93.77%
- **Summary:** Demonstrates strong performance in accuracy, precision, recall, and F1 score, making it effective for classification tasks.

### 4. Decision Tree Classifier
- **Accuracy:** 92.79%
- **Confusion Matrix:** [[2581, 325], [1144, 18297]]
- **Precision:** 94.49%
- **Recall:** 93.43%
- **F1 Score:** 93.76%
- **Summary:** Achieves a balance between accuracy, precision, recall, and F1 score.

Among the four models, the SVM Classifier performs the best with the highest accuracy, precision, recall, and F1 score. It is well-suited for identifying eligible borrowers due to its ability to create clear boundaries between classes in high-dimensional space and its lower susceptibility to overfitting.

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
### Function to select top 10 features based on MI scores for each target

| Top 10 features for EMI                      | Top 10 features for EligibleLoanAmount          | Top 10 features for PROI                       |
|----------------------------------------------|-------------------------------------------------|-----------------------------------------------|
| 1. TotalPaymentDue                           | 1. LP_CustomerPrincipalPayments                 | 1. LP_CustomerPrincipalPayments               |
| 2. LP_CustomerPrincipalPayments              | 2. TotalPaymentDue                              | 2. LP_GrossPrincipalLoss                      |
| 3. LoanOriginalAmount                        | 3. LP_CustomerPayments                          | 3. LP_NetPrincipalLoss                        |
| 4. Loan_tenure                               | 4. LP_GrossPrincipalLoss                        | 4. LoanFirstDefaultedCycleNumber              |
| 5. LP_CustomerPayments                       | 5. LP_NetPrincipalLoss                          | 5. LoanCurrentDaysDelinquent                  |
| 6. TotalAmount                               | 6. LP_ServiceFees                               | 6. TotalPaymentDue                            |
| 7. LP_ServiceFees                            | 7. Loan_tenure                                  | 7. MonthlyLoanPayment                         |
| 8. MonthlyLoanPayment                        | 8. LoanOriginalAmount                           | 8. LP_CustomerPayments                        |
| 9. InterestAmount                            | 9. TotalAmount                                  | 9. TotalAmount                                |
| 10. LP_GrossPrincipalLoss                    | 10. LoanFirstDefaultedCycleNumber               | 10. LoanOriginalAmount                        |
| 11. LP_NetPrincipalLoss                      | 11. LP_InterestandFees                          | 11. InterestAmount                            |
| 12. ROI                                      | 12. MonthlyLoanPayment                          | 12. Loan_tenure                               |
| 13. BorrowerRate                             | 13. LoanCurrentDaysDelinquent                   | 13. LP_ServiceFees                            |
| 14. LenderYield                              | 14. InterestAmount                              | 14. BorrowerAPR                               |
| 15. BorrowerAPR                              | 15. BorrowerAPR                                 | 15. ROI                                       |
| 16. LP_InterestandFees                       | 16. ROI                                         | 16. LP_InterestandFees                        |
| 17. LoanFirstDefaultedCycleNumber            | 17. LenderYield                                 | 17. LenderYield                               |
| 18. EstimatedLoss                            | 18. BorrowerRate                                | 18. BorrowerRate                              |
| 19. EstimatedEffectiveYield                  | 19. Closeddate_year                             | 19. EstimatedEffectiveYield                   |
| 20. ProsperRating (numeric)                  | 20. EstimatedEffectiveYield                     | 20. EstimatedReturn                           |

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




# 8. Build Pipeline
#### This pipeline is designed to handle both binary classification and multi-output regression tasks for loan prediction. The model aims to predict the    LoanStatus (classification) and three numerical targets: EMI, EligibleLoanAmount, and PROI (regression). 
###### a). Feature and Target Separation
        Features (X) are obtained by dropping EMI, EligibleLoanAmount, PROI, and LoanStatus columns from the DataFrame.
        The binary target (y_class) is LoanStatus.
        The multi-output targets (y_multi) are EMI, EligibleLoanAmount, and PROI.

###### b). Data Splitting
        The dataset is split into training and test sets with an 80-20 ratio.

###### c).Preprocessing Pipelines
        Numerical Features: Imputed with the mean and scaled using StandardScaler.
        Categorical Features: Imputed with the most frequent value and encoded using OneHotEncoder.

###### d).Classification Pipeline
 Model: XGBClassifier is used for binary classification.
    Steps:
        Preprocessing of features.
        Oversampling with SMOTE to handle class imbalance.
        Dimensionality reduction with PCA (30 components).
        Classification using XGBoost.
###### e).Regression Pipeline
        Model: MultiOutputRegressor wrapping RandomForestRegressor is used for multi-output regression.
        Steps:
        Preprocessing of features.
        Regression using the combined model.
###### f).Combined Pipeline

The CombinedPipeline class integrates both classification and regression pipelines.
Initialization: Takes the classification and regression pipelines as input.
    Fit Method:
        Trains the classification pipeline on X_train and y_train_class.
        Uses the trained classifier to predict LoanStatus on X_train, augmenting the data with PredictedLoanStatus.
        Trains the regression pipeline on the augmented data and y_train_multi.
    Predict Method:
        Predicts LoanStatus on X_test and augments the data with PredictedLoanStatus.
        Predicts EMI, EligibleLoanAmount, and PROI using the regression pipeline.
    Evaluate Method:
        Computes and prints Mean Squared Error (MSE) and R-squared (R²) score for each target variable on the test set.
###### g).Evaluation Metrics
      The model's performance on the test set is evaluated as follows:
EMI:
Mean Squared Error: 2151.89
        R² Score: 0.9986
EligibleLoanAmount:
        Mean Squared Error: 141812.66
        R² Score: 0.9172
PROI:
        Mean Squared Error: 0.000003
        R² Score: 0.9992 ```
These results indicate high accuracy for predicting EMI and PROI, with slightly lower but still strong performance for EligibleLoanAmount.

- Create an instance of CombinedPipeline:
``` combined_pipeline = CombinedPipeline(classification_pipeline, regression_pipeline) ```

-Train the pipeline:
```combined_pipeline.fit(X_train, y_train_class, y_train_multi)```

-Evaluate the pipeline:
```combined_pipeline.evaluate(X_test, y_test_multi)```
    
    
# 9. Save and Serialize the Model 
 ``` Save the combined pipeline to a file
     with open('combined_pipeline.pkl', 'wb') as f:
    pickle.dump(combined_pipeline, f) ```


# 9.  Deployment
### 9.1 Streamlit

The development process of the web application was Developed into the Streamlit.
Streamlit is an open source app framework in Python language. 
It helps us create web apps for data science and machine learning in a short time.
It is compatible with major Python libraries such as scikit-learn, NumPy, pandas, Matplotlib etc.

 ## Predict the LOAN APPLICATION with eligible borrower based on the three variable 
![Screenshot (474)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/6a59167e-5f6e-48b2-96e0-c7a8a7e69d4c)
![Screenshot (477)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/76a9e710-6686-4d55-8e70-e86f7e262c42)
![Screenshot (478)](https://github.com/Shraddhaass/TECHNOCOLABS_INTERNSHIP/assets/98949498/cc0631a4-5c10-4675-8145-6d20481d5343)
- The output of the application is EMI, Eligible loan amount and PROI.
