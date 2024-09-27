# Customer-Churn-Prediction

## 📄 Overview
This project aims to predict customer churn (i.e., whether a customer will leave or stay) using machine learning techniques. By analyzing a telecom customer dataset, we identify patterns in customer behavior and build predictive models to forecast churn. This project also provides recommendations for reducing churn based on key factors discovered during analysis.

## 📂 Dataset
The dataset used is the Telco Customer Churn Dataset from Kaggle. It includes information about:

- Customer demographics
- Services used by the customers
- Account details (e.g., tenure, monthly charges)
- Target variable: Churn (Yes/No)

The dataset can be found here: <p align="left"><a href="https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn/data"> Kaggle-Customer-Churn</a> </p>
## 🚀 Installation and Setup
Install the required libraries:
To run this project, you need to install the following libraries:

```python
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

```

Import the libraries in your Python script:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
```
## 🔍 Steps
### 1. Load the Dataset
```python
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.shape)
df.head()
```
### 2. Data Preprocessing

- Handle missing values in TotalCharges column and convert it to numeric.
- Encode categorical variables using Label Encoding for binary columns and One-Hot Encoding for multi-class columns.
- Scale numeric columns like tenure, MonthlyCharges, and TotalCharges.
  
```python
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.dropna(subset=['TotalCharges'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)
```

### 3. Exploratory Data Analysis (EDA)

We analyzed the churn distribution and calculated Cramér's V for categorical features to understand relationships in the data.

```python
import scipy.stats as stats

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod']

features = categorical_features + ['Churn']
cramers_results = pd.DataFrame(np.zeros((len(features), len(features))), columns=features, index=features)

for f1 in features:
    for f2 in features:
        cramers_results.loc[f1, f2] = cramers_v(df[f1], df[f2])

plt.figure(figsize=(12,10))
sns.heatmap(cramers_results, annot=True, cmap='coolwarm')
plt.title("Cramér's V Correlation Matrix")
plt.show()
```
![image](https://github.com/user-attachments/assets/9dced876-d8f0-4e6f-b46b-17f442172ec7)

### 4. Model Building

We built three models to predict churn:

* Logistic Regression
* Decision Tree
* Random Forest
  
SMOTE was applied to handle class imbalance in the dataset.

```python
X = df.drop('Churn', axis=1)
y = df['Churn']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train models
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

### 5. Model Evaluation
We evaluated the models using classification metrics and visualized the confusion matrices and ROC curves.

```python
# Random Forest Example
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# ROC Curve
y_pred_prob_rf = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf)
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

# AUC Score
auc_score_rf = roc_auc_score(y_test, y_pred_prob_rf)
print(f'AUC Score: {auc_score_rf:.4f}')
6. Hyperparameter Tuning
We used GridSearchCV to tune the hyperparameters of the Random Forest model.

python
Copy code
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Best Parameters: {grid_search.best_params_}')
7. Feature Importance
python
Copy code
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/a456e779-7937-46e7-9e18-6c807f57ed32)

Tuned Random Forest AUC Score: 0.9210

![image](https://github.com/user-attachments/assets/a2d5599c-3a4e-4b4b-8529-6f8c54dda8e8)

## 📊 Key Findings

* Important Features: Tenure, MonthlyCharges, and TotalCharges are among the most important features in predicting customer churn.
* Customers with shorter tenure and higher monthly charges are more likely to churn.
* Contract type and payment methods also play significant roles in churn prediction.

## 🛠️ Recommendations

* Offer incentives to customers with shorter tenure to encourage them to stay.
* Review pricing strategies for services with high monthly charges.
* Enhance service quality to address issues linked to higher churn rates.

## 📈 Real-Life Applications
This model can be used to predict churn across various industries such as telecom, banking, or retail. By identifying customers who are likely to leave, businesses can implement targeted retention strategies to reduce churn.
