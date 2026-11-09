# ✈️ Airline Flight Price Prediction (Airline_ML)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict airline flight prices based on various features such as the airline, journey date, source, destination, departure/arrival times, and the number of stops. 

The goal of this repository is to demonstrate robust data preprocessing, exploratory data analysis (EDA), feature engineering, and the training of multiple regression models to accurately forecast ticket prices.

## 🗂️ Project Structure

Based on the repository architecture, the project is organized as follows:

```text
Airline_ML/
│
├── Data/
│   ├── Airline1.csv                   # Original raw dataset
│   ├── Airline_Cleaned.csv            # Cleaned dataset after preprocessing
│   ├── Cleaned_Split_Data/            # Train/Test splits (X_train, y_train, etc.)
│   └── Transformed_data/              # Scaled and encoded data ready for modeling
│
├── Models/                            # Serialized ML models (.pkl)
│   ├── linear_model.pkl
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   └── elastic_net_model.pkl
│
├── Transformation_files/              # Saved preprocessors
│   └── transform.pkl                  # Scikit-learn ColumnTransformer pipeline
│
├── Jup_folder/                        # Jupyter notebooks containing the EDA and pipeline
└── README.md
```

## 🛠️ Machine Learning Pipeline

### 1. Data Cleaning & Preprocessing
* **Handling Missing Values:** Imputed missing values using the mode for categorical variables (Source, Destination, Route, Total_Stops) and the mean for numerical variables (Price).
* **Data Standardization:** Corrected spelling inconsistencies (e.g., merging 'New Delhi' and 'Delhi').
* **Feature Selection:** Dropped irrelevant columns (`ID`, `Additional_Info`, `Route`) that do not contribute to predictive power.

### 2. Feature Engineering
* **Duration Conversion:** Parsed the string-based `Duration` column (e.g., "2h 50m") and converted it into a continuous numerical feature (`Duration_Hours_Minutes`).
* **Datetime Extraction:** Extracted meaningful numerical features (`Day`, `Month`, `Year`, `Hour`, `Minute`) from raw `Date_of_Journey`, `Dep_Time`, and `Arrival_Time` strings.

### 3. Outlier Handling
* **IQR Method:** Utilized Seaborn boxplots to identify price outliers. Applied the Interquartile Range (IQR) method dynamically for *each individual airline* to calculate custom upper and lower fences, removing statistically significant anomalies without losing vital underlying trends.

### 4. Data Transformation
* **Scikit-Learn ColumnTransformer:** Created a robust, reusable transformation pipeline (`transform.pkl`).
* **Categorical Encoding:** Applied `OneHotEncoder` to categorical features (`Airline`, `Source`, `Destination`).
* **Numerical Scaling:** Applied `StandardScaler` to numerical features (time, duration, stops) to ensure optimal performance for distance-based and regularized algorithms.

### 5. Model Training & Evaluation
Multiple regression models were evaluated using 5-fold Cross-Validation via `GridSearchCV`. The models were assessed based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Adjusted R² Score.

* **Linear Regression:** Achieved an R² of ~0.71 on the test set.
* **Ridge Regression (L2):** Achieved an R² of ~0.708.
* **Lasso Regression (L1):** Evaluated for feature selection and regularization.
* **Elastic Net:** Evaluated to combine L1 and L2 penalties.

*(Note: Advanced ensemble models like Random Forest, XGBoost, and SVR are also part of the grid search space for future iteration).*

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/omkarbhat1995/Airline_ML.git
cd Airline_ML
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

**3. Run the Jupyter Notebooks:**
Navigate to the `Jup_folder/` to view the step-by-step EDA, data cleaning, and modeling processes.

**4. Load the Pre-trained Models:**
You can easily load the transformation pipeline and the best model to make predictions on new data:
```python
import joblib
import pandas as pd

# Load the transformer and model
transformer = joblib.load('Transformation_files/transform.pkl')
model = joblib.load('Models/ridge_model.pkl')

# Example prediction workflow
# X_new_transformed = transformer.transform(X_new)
# predictions = model.predict(X_new_transformed)
```

