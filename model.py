import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import r2_score # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Step 1: Data Preparation
# Load dataset
def dewpt(dataOP):
    data = pd.read_csv("data\JaipurFinalCleanData.csv")

    # Handling missing values if any
    data.dropna(inplace=True)

    # Splitting data into features and target variable
    X = data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1']]
    y = data['meandewptm_2']

    # Step 2: Model Training
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()
    #feed the training data
    model.fit(X_train, y_train)

    # Step 3: Model Evaluation
    # Predicting on the testing set
    y_pred = model.predict(X_test)

    # Evaluating model performance
    accuracy = r2_score(y_test, y_pred)

    # Step 4: Prediction
    # Assuming new_data contains the features (Temperature, Humidity, Pressure, AirSpeed, RainToday) for a new day
    new_data = pd.DataFrame([dataOP], columns=['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1'])
    prediction = model.predict(new_data)
    return prediction
def temp(dataOP):
    weather_data = pd.read_csv("data\JaipurFinalCleanData.csv")

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
    # print("Mean Squared Error:", mse)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)
    # print("R-squared:", r_squared)

    # Plot actual vs predicted values
    # plt.scatter(y_test, y_pred)
    # plt.xlabel("Actual Mean_Tempareture (mm)")
    # plt.ylabel("Predicted Mean_Tempareture (mm)")
    # plt.title("Actual vs Predicted Tempareture")
    # plt.show()
    new_data = pd.DataFrame([dataOP], columns=['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1'])

    # Making prediction for the new day
    prediction = model.predict(new_data)
    # print("Predicted Mean Tomorrow's Temperature:", prediction)
    return prediction
# print(dewpt([41,27,5,12,12,-2,1009,1000,0]))



def humidity(dataOP):
    weather_data = pd.read_csv('data\\JaipurFinalCleanData.csv')

# Select features (X) and target variable (y)
    X = weather_data[['maxtempm', 'mintempm', 'maxhumidity_1','minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1']]  
    y = weather_data['maxhumidity_2']  

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
    # plt.scatter(y_test, y_pred)
    # plt.xlabel("Actual Maximum_Humidity (g/m^3)")
    # plt.ylabel("Predicted Maximum_Humidity (g/m^3)")
    # plt.title("Actual vs Predicted Humidity")
    # plt.show()
    new_data = pd.DataFrame([dataOP], columns=['maxtempm', 'mintempm', 'maxhumidity_1','minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1'])

    # Making prediction for the new day
    prediction = model.predict(new_data)
    return prediction

data = [41,27,5,12,12,-2,1009,1000,0]
K=temp(data)
print(K)
D=dewpt(data)
print(D)
H=humidity(data)
print(H)