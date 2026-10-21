
# Week 1: Python for Machine Learning and Data Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 2: Explore the dataset
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# Step 3: Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Step 4: Remove duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Step 5: Feature selection
# Remove columns that are not needed for this basic analysis
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Step 6: Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Step 7: Encode categorical variables
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

# Step 8: Normalize numerical features
scaler = MinMaxScaler()

columns_to_scale = ["Age", "Fare", "SibSp", "Parch"]

df[columns_to_scale] = scaler.fit_transform(
    df[columns_to_scale]
)

# Step 9: Exploratory data analysis (EDA)
print("\nSurvival count:")
print(df["Survived"].value_counts())

print("\nCorrelation with survival:")
print(df.corr(numeric_only=True)["Survived"].sort_values(
    ascending=False
))

# Step 10: Check the cleaned dataset
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal dataset shape:", df.shape)
print("\nCleaned dataset preview:")
print(df.head())

# Step 11: Save the cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")
