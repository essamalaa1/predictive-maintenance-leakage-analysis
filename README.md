# Predictive Maintenance: Leakage Analysis & Classification

This project explores the task of predicting machine failures using a predictive maintenance dataset. A key focus is on identifying and understanding the impact of data leakage, particularly from the `Failure Type` feature. The project also demonstrates data preprocessing, exploratory data analysis (EDA), model building (Logistic Regression, Naive Bayes), and evaluation, including the use of SMOTE for handling imbalanced data.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Key Analyses Performed](#key-analyses-performed)
    *   [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    *   [Data Leakage Detection](#data-leakage-detection)
    *   [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
    *   [Model Building & Evaluation](#model-building--evaluation)
4.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Dependencies](#dependencies)
    *   [Data](#data)
5.  [Usage](#usage)
6.  [Results Summary](#results-summary)
    *   [Models without Leakage](#models-without-leakage)
    *   [Models with Leakage](#models-with-leakage)
    *   [Impact of SMOTE](#impact-of-smote)
7.  [Leakage Detection Techniques Discussed](#leakage-detection-techniques-discussed)
8.  [Code Structure Highlights](#code-structure-highlights)
9.  [Potential Future Work](#potential-future-work)

## Project Overview
The primary goal is to build a classification model to predict machine failures (`Target` variable). The project emphasizes:
*   Thorough EDA to understand data characteristics.
*   Identifying data leakage, primarily from the `Failure Type` column, which is a post-event indicator of the failure.
*   Comparing model performance with and without the leaky feature to demonstrate its impact.
*   Addressing class imbalance in the target variable using the SMOTE (Synthetic Minority Over-sampling Technique).
*   Evaluating models based on accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset
The project uses the `predictive_maintenance.csv` dataset. The key features include:
*   **UDI**: Unique identifier.
*   **Product ID**: Product identifier.
*   **Type**: Quality of the product (L, M, H).
*   **Air temperature [K]**: Air temperature in Kelvin.
*   **Process temperature [K]**: Process temperature in Kelvin.
*   **Rotational speed [rpm]**: Rotational speed of the tool.
*   **Torque [Nm]**: Torque applied by the tool.
*   **Tool wear [min]**: Tool wear in minutes.
*   **Target**: Binary indicator of machine failure (0 = No Failure, 1 = Failure).
*   **Failure Type**: Specific type of failure (e.g., "No Failure", "Power Failure", "Tool Wear Failure"). This is the primary source of identified leakage.

## Key Analyses Performed

### Exploratory Data Analysis (EDA)
*   Loaded the dataset and examined its structure (`df.head()`, `df.info()`).
*   Generated summary statistics (`df.describe()`).
*   Checked for missing values (`df.isnull().sum()`).
*   Analyzed the distribution of the `Target` variable and `Failure Type`.
*   Investigated the relationship between `Target` and `Failure Type` using crosstabs.
*   Examined the correlation between `Type` and `Target`.
*   Visualized correlations among numerical features using a heatmap.

### Data Leakage Detection
*   **Correlation and Crosstab Analysis**: Showed a near-perfect mapping between `Failure Type` and `Target`, indicating severe data leakage.
*   **Random Forest Feature Importance**:
    *   When `Failure Type` (one-hot encoded) was included, it dominated feature importances, confirming leakage.
    *   When `Failure Type` was excluded, `Torque [Nm]` became the most important feature.

### Feature Engineering & Preprocessing
*   Dropped non-predictive identifiers (`UDI`, `Product ID`).
*   Dropped features with low correlation or redundancy based on EDA (e.g., `Process temperature [K]`, `Rotational speed [rpm]`).
*   **Clean Dataset**: Excluded `Failure Type` to prevent leakage.
*   **Leaky Dataset**: Included `Failure Type` to demonstrate its impact.
*   One-hot encoded categorical features (`Type`, `Failure Type` for the leaky model).
*   Split data into training and testing sets (80/20 split, stratified by `Target`).
*   Standardized numerical features using `StandardScaler` (fit on training data, transformed on both train and test to prevent leakage from the test set).
*   Applied SMOTE to the training data to address class imbalance for the `Target` variable.

### Model Building & Evaluation
Two primary classification models were trained and evaluated:
*   **Logistic Regression**
*   **Gaussian Naive Bayes**

These models were trained on four dataset variations:
1.  Cleaned data (no leakage, original imbalance).
2.  Cleaned data with SMOTE applied to the training set.
3.  Leaky data (including `Failure Type`, original imbalance).
4.  Leaky data with SMOTE applied to the training set.

Evaluation metrics included accuracy, precision, recall, F1-score, confusion matrix, and classification report.

## Setup and Installation

### Prerequisites
*   Python (3.8+ recommended)
*   pip (Python package installer)

### Dependencies
The project uses the following Python libraries. You can install them using pip:
```bash
pip install pandas matplotlib seaborn tabulate scikit-learn imbalanced-learn

Or create a requirements.txt file:

pandas
matplotlib
seaborn
tabulate
scikit-learn
imbalanced-learn

And install with:

pip install -r requirements.txt