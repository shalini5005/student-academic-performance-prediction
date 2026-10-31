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
- Root Mean Squared Error (RMSE)
- R-squared (R²)

K-Means clustering was explored separately because it is an unsupervised learning algorithm.

## Project Structure

```text
Student_Academic_Performance_CA2/
│
├── main.py
├── student_grade_ses_dataset.csv
└── README.md
```

## How to Run the Project

1. Clone this repository:

```bash
git clone https://github.com/shalini5005/student-academic-performance-prediction.git
```

2. Navigate to the project folder:

```bash
cd student-academic-performance-prediction
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the Python file:

```bash
python main.py
```

## Key Learning Outcomes

- Learned the basics of predictive analytics and machine learning.
- Practiced data cleaning and preprocessing.
- Explored student academic data through visualizations.
- Implemented regression and clustering algorithms.
- Compared regression model performance using evaluation metrics.

## Conclusion

This project provided practical experience in applying machine learning techniques to student academic data. Among the regression models tested, Linear Regression achieved the best R² score in the initial model comparison. The clustering model also helped explore groups of students with different academic characteristics.

---

**Project Type:** Academic / Machine Learning Project  
**Language:** Python