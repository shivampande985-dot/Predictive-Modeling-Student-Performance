# ==========================================
# PREDICTIVE MODELING USING MACHINE LEARNING
# Student Performance Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("student-mat.csv", sep=';')

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# DATA PREPROCESSING
# ==========================================

# Convert categorical columns into numerical values
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and Target Variable
X = df_encoded.drop('G3', axis=1)
y = df_encoded['G3']

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# LINEAR REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n===== LINEAR REGRESSION =====")
print("R² Score:", r2_score(y_test, y_pred_lr))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_lr))

# ==========================================
# DECISION TREE MODEL
# ==========================================

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n===== DECISION TREE =====")
print("R² Score:", r2_score(y_test, y_pred_dt))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_dt))

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("R² Score:", r2_score(y_test, y_pred_rf))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))

# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance.head(10),
    x='Importance',
    y='Feature'
)

plt.title("Top 10 Important Features")
plt.show()

# ==========================================
# MODEL COMPARISON
# ==========================================

models = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest"
]

scores = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_dt),
    r2_score(y_test, y_pred_rf)
]

plt.figure(figsize=(8, 5))
plt.bar(models, scores)

plt.title("Model Comparison")
plt.ylabel("R² Score")
plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("\n========== CONCLUSION ==========")
print("1. Three machine learning models were trained.")
print("2. Random Forest generally provides the best accuracy.")
print("3. Previous grades (G1, G2) are strong predictors of G3.")
print("4. Study time and absences also influence performance.")
print("5. Predictive modeling can help identify students who may need academic support.")

print("\nProject Completed Successfully!")
