# 1. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import joblib


# 2. Load dataset
df = pd.read_csv('C:/Users/APRAJ/OneDrive/Desktop/anaconda/diabets.csv')
print(df.head())

# 3. Data preprocessing
print(df.isnull().sum())  # check for missing values
print(df.describe())      # summary stats

# Optional: Replace 0s in Glucose, BloodPressure, etc. with NaN and impute
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.NaN)
df.fillna(df.mean(), inplace=True)

# 4. Feature scaling
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 6. Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')  # This saves your model
joblib.dump(scaler, 'scaler.pkl')         # This saves the StandardScaler
