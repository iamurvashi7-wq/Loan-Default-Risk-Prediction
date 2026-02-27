# Loan-Default-Risk-Prediction

## ⇨Project Overview :-
This project focuses on predicting loan default risk using machine learning techniques. 
The objective is to analyze borrower financial and demographic information to determine the likelihood of loan default.
Loan default prediction is an important application in financial institutions, as incorrect credit decisions can lead to financial losses. 
The developed model is intended to function as a decision-support tool rather than a fully automated approval system.

## ⇨Dataset Description
The dataset used in this project contains borrower-level and loan-related attributes. 

### Key Features

#### Borrower Information :- 

•Age

•Income

•Education

•Employment Type

•Marital Status

#### Credit and Financial Indicators :-

•CreditScore

•Debt-to-Income Ratio (DTIRatio)

•Number of Credit Lines

•HasMortgage

•HasDependents

•HasCoSigner

#### Loan Characteristics :-

•LoanAmount

•InterestRate

•LoanTerm

•LoanPurpose

#### Target Variable :-

•Default :- •0 = No Default  •1 = Default

The dataset contains 255,347 records and 18 features.

## ⇨Data Preprocessing and Feature Engineering
The following preprocessing steps were applied:

•Removed irrelevant column (LoanID)

•Checked and confirmed no missing values

•Converted binary variables (Yes/No) into numerical format (1/0)

•Applied one-hot encoding to categorical variables

•Created derived features:- Loan_to_income ratio , Income_per_creditline , Employment_Years

•Performed 8-bin grouping on selected continuous variables for segmentation analysis

•Applied train-test split for model training

## ⇨Exploratory Data Analysis (EDA)
EDA was conducted to understand borrower behavior and default patterns. Key findings:

•Default rate decreases as income increases.

•Default rate decreases as credit score increases.

•Loan-to-income ratio shows positive relationship with default risk.

•Default rate increases with higher loan amounts.

•Dataset shows class imbalance, with fewer default cases compared to non-defaults.

Correlation analysis and visualizations were used to support these insights.

## ⇨Modeling Approach
The problem was treated as a binary classification task.

### Model Used:
•Logistic Regression (baseline model)

### Logistic Regression was selected because:

•It is interpretable

•It provides probability-based predictions

•It is widely used in credit risk modeling

## ⇨Model Evaluation
### The model was evaluated using:

•Accuracy

•Precision

•Recall

•Confusion Matrix

### Performance Summary:

•Accuracy: 69%

•Precision: 23%

•Recall: 69%

The model demonstrates good recall, meaning it successfully identifies a significant portion of defaulters.
However, precision is relatively low, indicating that many safe borrowers are classified as risky.
From a risk management perspective, minimizing false negatives (approving risky borrowers) is more critical than minimizing false positives.

## ⇨Key Findings

•Income and credit score have negative relationships with default probability.

•Loan amount has a positive relationship with default risk.

•No single variable strongly predicts default independently.

•Multiple financial indicators combined improve predictive performance.

## ⇨Model Limitations

•Dataset shows class imbalance.

•Model generates a high number of false positives.

•Economic conditions and external factors are not included.

•Predictions should not replace human credit assessment.

