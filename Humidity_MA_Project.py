# Importing necessary libraries
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the weather data
weather_data = pd.read_csv('data\JaipurFinalCleanData.csv')

# Select features (X) and target variable (y)
X = weather_data[['meantempm','maxtempm', 'mintempm',  'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1']]  
y = weather_data['maxhumidity_1']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train) 

# Make predictions
y_pred = model.predict(X_test)
print("Maximum_Humidity : ",y_pred)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Maximum_Humidity (g/m^3)")
plt.ylabel("Predicted Maximum_Humidity (g/m^3)")
plt.title("Actual vs Predicted Humidity")
plt.show()
new_data = pd.DataFrame([[28,34,21,13,16,6,1011,1003,5]], columns=['meantempm','maxtempm', 'mintempm', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1'])

# Making prediction for the new day
prediction = model.predict(new_data)
print("Predicted Maximum_Humidity:", prediction)
