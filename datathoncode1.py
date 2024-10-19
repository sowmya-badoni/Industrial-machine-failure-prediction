# Data Preprocessing

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Load the dataset
data = pd.read_csv("C:/Users/admin/OneDrive/Desktop/IBM DATATHON/predictive_maintenance_dataset.csv")

# Check the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Handling missing values: Drop rows or fill missing data (depending on the data)
data.dropna(inplace=True)

# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract useful components from the date, such as year, month, and day
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Optionally, drop the original 'date' column if it's no longer needed
data = data.drop(columns=['date'])

# Check if all columns are numeric
print(data.dtypes)

# Identify non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)

# If you want to drop non-numeric columns
data.drop(columns=non_numeric_cols, inplace=True)

# Drop the target column ('failure') before scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['failure']))

# Convert scaled data back to DataFrame and re-add the 'failure' column
data_scaled = pd.DataFrame(scaled_data, columns=data.columns[:-1])
data_scaled['failure'] = data['failure'].values  # Assuming 'failure' is the target

# Display the cleaned and scaled data
print(data_scaled.head())

# Exploratory Data Analysis (EDA)

# Plot a sample of features (first 10 features)
num_features_to_plot = 10  # Adjust as needed
columns_to_plot = data_scaled.columns[:-1][:num_features_to_plot]  # Exclude 'failure'

plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_plot):
    plt.subplot(5, 2, i + 1)
    plt.hist(data_scaled[col], bins=20)
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Box plots for selected features
plt.figure(figsize=(15, 10))
sns.boxplot(data=data_scaled[columns_to_plot])
plt.title('Box Plots of Selected Features')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_scaled.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Features')
plt.show()

# Count plot of the target variable (failure vs no failure)
sns.countplot(x='failure', data=data_scaled)
plt.title('Failure vs Non-Failure Distribution')
plt.show()

# Model Selection

# Split data into features (X) and target (y)
X = data_scaled.drop('failure', axis=1)  # Features
y = data_scaled['failure']  # Target (failure or not)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model: Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Model Evaluation

# Use SMOTE to oversample the minority class (failures)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model on resampled data
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model: Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-Validation for Stability

# Perform cross-validation on the resampled data (using 5 folds)
cv_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=5)

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Hyperparameter Tuning

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with RandomForest
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Print classification report of the best model
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))

# XGBoost Classifier

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print("Classification Report for XGBoost:")
print(classification_report(y_test, y_pred_xgb))
