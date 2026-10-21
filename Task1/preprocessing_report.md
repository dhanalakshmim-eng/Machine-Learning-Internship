# Week 1 Task: Python for Machine Learning and Data Preprocessing

## 1. Objective

The objective of this task is to understand the fundamentals of machine learning using Python and learn how to load, explore, clean, and preprocess a dataset using Pandas, NumPy, and Scikit-learn.

## 2. Dataset Description

- **Dataset Name:** Titanic - Machine Learning from Disaster
- **Source:** Kaggle
- **File Name:** `train.csv`
- **Data Type:** Structured tabular data
- **Target Variable:** `Survived`

The dataset contains passenger information, including age, sex, passenger class, fare, and survival status. It is used to demonstrate data preprocessing and exploratory data analysis.

## 3. Tools and Libraries Used

- **Python:** Programming language used for data processing.
- **Pandas:** Used to load, inspect, clean, and manipulate the dataset.
- **NumPy:** Used for numerical data operations.
- **Scikit-learn:** Used for feature normalization with `MinMaxScaler`.
- **VS Code:** Used to write and execute the Python program.

## 4. Preprocessing Steps

### Step 1: Data Loading

Loaded the Titanic dataset using Pandas `read_csv()`.

### Step 2: Data Exploration

Used `head()`, `info()`, and `describe()` to inspect sample records, column data types, non-null counts, and numerical statistics.

### Step 3: Missing Value Identification

Used `isnull().sum()` to identify columns containing missing values.

### Step 4: Duplicate Removal

Checked for duplicate rows using `duplicated()` and removed them using `drop_duplicates()`.

### Step 5: Feature Selection

Removed `PassengerId`, `Name`, `Ticket`, and `Cabin` for this basic preprocessing exercise. These columns were excluded to simplify the analysis.

### Step 6: Handling Missing Values

- Replaced missing values in `Age` with the median age.
- Replaced missing values in `Embarked` with the most frequent category (mode).

### Step 7: Encoding Categorical Variables

Applied one-hot encoding using `pd.get_dummies()` to convert the categorical columns `Sex` and `Embarked` into numerical features.

### Step 8: Normalization

Used Scikit-learn's `MinMaxScaler` to scale `Age`, `Fare`, `SibSp`, and `Parch` to a range between 0 and 1.

### Step 9: Exploratory Data Analysis (EDA)

Examined survival counts and calculated correlations between numerical features and the `Survived` target variable.

### Step 10: Exporting the Cleaned Dataset

Saved the processed data as `cleaned_titanic.csv` using Pandas `to_csv()`.

## 5. Results

- Successfully loaded and explored the Titanic dataset.
- Identified and handled missing values in the selected columns.
- Checked for and removed duplicate rows.
- Converted categorical features into numerical form.
- Normalized selected numerical features.
- Generated a cleaned dataset for further analysis.

**Output File:** `cleaned_titanic.csv`

## 6. Conclusion

This task provided practical experience in preparing a dataset for machine learning using Python. It covered data loading, data inspection, missing value handling, duplicate removal, feature selection, categorical encoding, normalization, and exploratory data analysis.

The resulting dataset can be used for further analysis and as a starting point for machine learning experiments.

## 7. Project Structure

```text
Machine-Learning-Internship/
└── Task1/
    ├── preprocessing.py
    ├── train.csv
    ├── cleaned_titanic.csv
    └── preprocessing_report.md
```
