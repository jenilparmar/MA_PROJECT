# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the weather data
weather_data = pd.read_csv('data\JaipurFinalCleanData.csv')

# Select features (X) and target variable (y)
X = weather_data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1']]  
y = weather_data['meantempm']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Mean_Tempareture (mm)")
plt.ylabel("Predicted Mean_Tempareture (mm)")
plt.title("Actual vs Predicted Tempareture")
plt.show()
new_data = pd.DataFrame([[41,27,5,12,12,-2,1009,1000,0]], columns=['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1'])

# Making prediction for the new day
prediction = model.predict(new_data)
print("Predicted Mean Tomorrow's Temperature:", prediction)