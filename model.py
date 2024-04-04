import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Data Preparation
# Load dataset
data = pd.read_csv("sh1.csv")

# Handling missing values if any
data.dropna(inplace=True)

# Splitting data into features and target variable
X = data[['Temperature', 'Humidity', 'Pressure', 'AirSpeed', 'RainToday']]
print(X)
y = data['RainfallTomorrow']
print(y)

# Step 2: Model Training
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
print(X_test,X_train)

# Initialize the linear regression model
model = LinearRegression()
#feed the training data
model.fit(X_train, y_train)

# Step 3: Model Evaluation
# Predicting on the testing set
y_pred = model.predict(X_test)

# Evaluating model performance
accuracy = r2_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Regression Coefficients:", model.coef_)
# Step 4: Prediction
# Assuming new_data contains the features (Temperature, Humidity, Pressure, AirSpeed, RainToday) for a new day
new_data = pd.DataFrame([[20.1,57,1011.7,33,0]], columns=['Temperature', 'Humidity', 'Pressure', 'AirSpeed', 'RainToday'])

# Making prediction for the new day
prediction = model.predict(new_data)
print("Predicted Rainfall Tomorrow:", prediction)
