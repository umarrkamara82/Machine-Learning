import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\user\Desktop\Umarr- Dr. AJ\ML\student-mat.csv", delimiter=';')

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target variable
X = df.drop(columns=['G3'])  # Predicting final grade G3
y = df['G3']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual G3 Grades")
plt.ylabel("Predicted G3 Grades")
plt.title("Actual vs Predicted Final Grades")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.show()

# Feature Importance Visualization
plt.figure(figsize=(12, 6))
feature_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['G3']).columns)
feature_importances.nlargest(10).plot(kind='barh', colormap='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features in Predicting G3")
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Actual vs Predicted Boxplot
plt.figure(figsize=(10, 6))
data = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
sns.boxplot(data=data)
plt.title("Comparison of Actual and Predicted G3 Grades")
plt.show()

# Display Feature Importances as a Table
feature_imp_df = feature_importances.nlargest(10).reset_index()
feature_imp_df.columns = ["Feature", "Importance"]
print("Top 10 Features in Predicting G3:\n", feature_imp_df)
