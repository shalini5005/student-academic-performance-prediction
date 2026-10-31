# Student Academic Performance Prediction

A machine learning project that analyzes student academic data and explores factors associated with academic performance using regression and clustering techniques.

## About the Project

The aim of this project is to analyze student academic performance using demographic, family, and educational information. It uses machine learning techniques to predict students' final grade percentages and identify groups of students with similar academic characteristics.

## Dataset

- **Dataset:** Student Grade and Socioeconomic Status
- **Records after removing duplicates:** 12,000
- **Original columns:** 29
- **Target variable:** `final_grade_pct`

The dataset contains information such as:
- Student age and gender
- Family and socioeconomic background
- Study hours per week
- Attendance rate
- Previous year grades
- Final grade percentage

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed duplicate records.
2. Converted the survey date column into datetime format.
3. Handled missing values using median and mode.
4. Removed irrelevant and target-related columns from the input features.
5. Converted categorical features into numerical form using one-hot encoding.
6. Applied feature scaling where required.

## Machine Learning Models

Five machine learning techniques were implemented:

1. **Linear Regression** – Predicts final grade percentages using a linear relationship.
2. **Polynomial Regression** – Captures possible nonlinear relationships between features and grades.
3. **KNN Regression** – Predicts grades using similarities between students.
4. **Decision Tree Regression** – Predicts grades using decision-based rules.
5. **K-Means Clustering** – Groups students based on study hours, attendance, and previous-year grades.

## Model Evaluation

The regression models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squ