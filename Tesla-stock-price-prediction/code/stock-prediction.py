import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for better readability

# Load the dataset
# The dataset contains historical stock prices for Tesla (TSLA)
df = pd.read_csv("C:/Users/swapn/Data Science & ML/Applied Data Science/Project -3/TSLA.csv")

# Display the first few rows and statistical summary of the dataset
print(df.head())  # Quick preview of the dataset
print(df.describe())  # Statistical insights (mean, std, min, max, etc.)

# Visualization: Plot Tesla's closing price over time
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close Price', fontsize=15)
plt.ylabel('Price in dollars')
plt.show()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Drop the "Adj Close" column as it won't be used for modeling
df = df.drop(['Adj Close'], axis=1)

# Check for null values in the dataset
print("Null values in each column:\n", df.isnull().sum())

# Feature list for visualization
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Plot histograms for key features to analyze their distributions
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True)  # Histogram with a density plot
plt.show()

# Plot boxplots for key features to identify outliers
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(x=df[col])
plt.show()

# --- Date Processing ---
# Convert the Date column to datetime format for extracting day, month, and year
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle errors gracefully
df = df.dropna(subset=['Date'])  # Drop rows with invalid dates

# Extract day, month, and year from the Date column
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Mark rows where the month is the end of a fiscal quarter
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Display the modified DataFrame to confirm these changes
print(df.head())

# Annual grouping: Find yearly averages for key features
data_grouped = df.drop('Date', axis=1).groupby('year').mean()

# Visualization: Bar plots to show yearly trends in stock prices
plt.subplots(figsize=(20, 10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    data_grouped[col].plot.bar()
    plt.title(f'Yearly average {col}')
plt.show()

# --- Feature Engineering ---
# Create additional features to capture relationships
df['open-close'] = df['Open'] - df['Close']  # Difference between opening and closing prices
df['low-high'] = df['Low'] - df['High']  # Difference between low and high prices

# Target variable: Predict whether the stock price increases the next day
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Plot the distribution of the target variable
plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%', startangle=90)
plt.title('Target Distribution')
plt.show()

# --- Correlation Analysis ---
# Generate a heatmap to visualize feature correlations
plt.figure(figsize=(10, 10))
sb.heatmap(df.drop('Date', axis=1).corr(), annot=True, cbar=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# --- Model Preparation ---
# Select features and target for model training
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Normalize features using StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(f"Training Set Shape: {X_train.shape}, Validation Set Shape: {X_valid.shape}")

# --- Model Training and Evaluation ---
# Define models to train: Logistic Regression, SVM, and XGBoost
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

# Train each model and evaluate performance
for model in models:
    model.fit(X_train, Y_train)  # Fit the model
    print(f'{model} :')
    print('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, model.predict_proba(X_train)[:, 1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, model.predict_proba(X_valid)[:, 1]))
    print()

# Plot confusion matrix for the best-performing model (Logistic Regression)
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
