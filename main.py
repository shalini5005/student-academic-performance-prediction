import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler

####################### STEP 1: LOAD THE DATASET ######################

df = pd.read_csv("student_grade_ses_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Display first five records
print("\nFirst five records:")
print(df.head())

# Display last five records
print("\nLast five records:")
print(df.tail())

# Display dataset information
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check duplicate records
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())


################# STEP 2: DATA CLEANING ####################

# Remove duplicate records
print("\nDuplicates before cleaning:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after cleaning:", df.duplicated().sum())

# Convert survey_date into datetime format
df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")

# Fill missing numerical values using median
numerical_columns = [
    "annual_family_income_inr",
    "distance_to_school_km",
    "attendance_rate_pct",
    "previous_year_grade_pct"
]

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing binary internet-access values using mode
df["has_internet_access"] = df["has_internet_access"].fillna(df["has_internet_access"].mode()[0])

# Fill missing categorical values using mode
categorical_columns = [
    "father_education_level",
    "mother_education_level",
    "father_occupation",
    "mother_occupation"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Display final dataset information
print("\nDataset shape after cleaning:", df.shape)
print("\nFinal dataset information:")
df.info()



######################## STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ################################

# Basic statistics of final grades
print("\nFinal Grade Statistics:")
print(df["final_grade_pct"].describe())

# Average final grade by gender
print("\nAverage Final Grade by Gender:")
print(df.groupby("gender")["final_grade_pct"].mean())

# Average final grade by school type
print("\nAverage Final Grade by School Type:")
print(df.groupby("school_type")["final_grade_pct"].mean())

# Histogram of final grades
plt.figure(figsize=(8, 5))
plt.hist(df["final_grade_pct"], bins=30, edgecolor="black")
plt.title("Distribution of Students' Final Grades")
plt.xlabel("Final Grade (%)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Study hours vs final grades
plt.figure(figsize=(8, 5))
plt.scatter(df["study_hours_per_week"], df["final_grade_pct"], alpha=0.4)
plt.title("Study Hours vs Final Grades")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Grade (%)")
plt.tight_layout()
plt.show()

# Attendance vs final grades
plt.figure(figsize=(8, 5))
plt.scatter(df["attendance_rate_pct"], df["final_grade_pct"], alpha=0.4)
plt.title("Attendance vs Final Grades")
plt.xlabel("Attendance Rate (%)")
plt.ylabel("Final Grade (%)")
plt.tight_layout()
plt.show()

# Correlation heatmap
numerical_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()


# STEP 4: DATA PREPROCESSING

# Separate input features (X) and target variable (y)

X = df.drop(columns=["student_id", "survey_date", "academic_year", "final_grade_pct", "grade_class", "at_risk_flag"])
y = df["final_grade_pct"]

print("\nInput features:")
print(X.head())

print("\nTarget variable:")
print(y.head())

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)


# Identify numerical and categorical columns

numerical_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

print("\nNumerical columns:")
print(list(numerical_columns))

print("\nCategorical columns:")
print(list(categorical_columns))


# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


####### Convert categorical columns into numerical columns

X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, dtype=int)

X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, dtype=int)

### Make sure both datasets have the same columns
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

print("\nTraining data after encoding:")
print(X_train_encoded.head())

print("\nTesting data after encoding:")
print(X_test_encoded.head())

print("\nShape after encoding:")
print("Training:", X_train_encoded.shape)
print("Testing:", X_test_encoded.shape)

#### Create a scaler
# Normalization for KNN

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

######################## MODEL 1: LINEAR REGRESSION ########################

# Create the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train_encoded, y_train)

# Predict final grades
lr_predictions = lr_model.predict(X_test_encoded)

# Evaluate the model
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_predictions)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

################ MODEL 2: POLYNOMIAL REGRESSION ################

# Polynomial transformation
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_encoded)
X_test_poly = poly.transform(X_test_encoded)

# Create model
poly_model = LinearRegression()

# Train model
poly_model.fit(X_train_poly, y_train)

# Prediction
poly_predictions = poly_model.predict(X_test_poly)

# Evaluation
poly_mse = mean_squared_error(y_test, poly_predictions)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, poly_predictions)
poly_mae = mean_absolute_error(y_test, poly_predictions)

print("\nPOLYNOMIAL REGRESSION RESULTS")
print("MSE:", poly_mse)
print("RMSE:", poly_rmse)
print("R2 Score:", poly_r2)
print("MAE:", poly_mae)

######################## MODEL 3: KNN REGRESSION ########################

# Create the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the model
knn_model.fit(X_train_scaled, y_train)

# Predict final grades
knn_predictions = knn_model.predict(X_test_scaled)

# Evaluate the model
knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, knn_predictions)

print("\nKNN REGRESSION RESULTS")
print("MAE:", knn_mae)
print("MSE:", knn_mse)
print("RMSE:", knn_rmse)
print("R2 Score:", knn_r2)

################ MODEL 4: DECISION TREE REGRESSION ################

# Create model
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)

# Train model
dt_model.fit(X_train_encoded, y_train)

# Prediction
dt_predictions = dt_model.predict(X_test_encoded)

# Evaluation
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)

print("\nDECISION TREE RESULTS")
print("MSE:", dt_mse)
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)
print("MAE:", dt_mae)

# Visualize Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(dt_model, feature_names=X_train_encoded.columns, filled=True, rounded=True, max_depth=3, fontsize=6)
plt.title("Decision Tree Regression")
plt.tight_layout()
plt.show()

################ MODEL 5: K-MEANS CLUSTERING ################

# Select features
cluster_data = df[[
    "study_hours_per_week",
    "attendance_rate_pct",
    "previous_year_grade_pct"
]]

# Normalize data
scaler = MinMaxScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit model
clusters = kmeans.fit_predict(cluster_data_scaled)

# Add cluster column
df["cluster"] = clusters

# Display results
print("\nK-MEANS RESULTS")
print(df["cluster"].value_counts())

print("\nCluster Average:")
print(df.groupby("cluster")[[
    "study_hours_per_week",
    "attendance_rate_pct",
    "previous_year_grade_pct",
    "final_grade_pct"
]].mean())

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="study_hours_per_week", y="previous_year_grade_pct", hue="cluster")
plt.title("K-Means Clustering")
plt.xlabel("Study Hours Per Week")
plt.ylabel("Previous Year Grade")
plt.show()