# Customer Churn Prediction 📊

This project implements a complete machine learning pipeline to predict customer churn for a telecommunications company. It addresses class imbalance using SMOTE and deploys an optimized Random Forest Classifier.

## 📁 Project Structure

* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Raw dataset containing client information.
* `customer_churn_model.pkl`: Serialized Random Forest model along with selected features list.
* `encoders.pkl`: Fitted `LabelEncoder` objects for categorical feature transformations.
* `README.md`: Project description and documentation.

## 🚀 Features

- **Data Cleaning**: Handled missing `TotalCharges` values and removed uninformative columns (`customerID`).
- **Exploratory Data Analysis**: Deep analysis of numerical distributions, correlations, and categorical impacts on churn rate.
- **Handling Class Imbalance**: Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance target categories for stable training.
- **Robust Evaluation**: Employed 5-Fold Cross-Validation, resulting in a **Random Forest accuracy of ~84%** during validation.

## 📈 Model Performance on Test Set

* **Overall Accuracy**: ~77.85%
* **Classification Breakdown**:
  - **Class 0 (No Churn)**: Precision 0.85 | Recall 0.85
  - **Class 1 (Churn)**: Precision 0.58 | Recall 0.59

## ⚙️ How to Run the Predictive System

```python
import pandas as pd
import pickle

# 1. Load the model & encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    loaded_model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# 2. Input your customer data
customer_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

customer_data_df = pd.DataFrame([customer_data])

# 3. Transform categorical features
for col, encoder in encoders.items():
    customer_data_df[col] = encoder.transform(customer_data_df[col])

# 4. Predict Churn
prediction = loaded_model.predict(customer_data_df)
print("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
```
