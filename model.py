import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask ,render_template,request
app = Flask(__name__)
def dewpt(dataOP):
    data = pd.read_csv("data\JaipurFinalCleanData.csv")

    # Handling missing values if any
    data.dropna(inplace=True)

   # Splitting data into features and target variable
    X = data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1']]
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
    new_data = pd.DataFrame([dataOP], columns=['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1'])
    prediction = model.predict(new_data)
    return prediction
def temp(dataOP):
    weather_data = pd.read_csv("data\JaipurFinalCleanData.csv")

    X = weather_data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1']]  
    y = weather_data['meantempm']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
  
    r_squared = r2_score(y_test, y_pred)
    
    new_data = pd.DataFrame([dataOP], columns=['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1'])

    # Making prediction for the new day
    prediction = model.predict(new_data)
    # print("Predicted Mean Tomorrow's Temperature:", prediction)
    return prediction
def rainfall(data):
    weather_data = pd.read_csv("data\JaipurFinalCleanData _NextDay.csv")
    X = weather_data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1']]  
    y = weather_data['precipm_2_B_nextday']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Precipitation on test data : ",y_pred)

    new_data = pd.DataFrame([data], columns=['maxtempm', 'mintempm', 'maxhumidity_1','minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1'])

    # Making prediction for the new day
    prediction = model.predict(new_data)
    return prediction

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the input values from the form
        maxtempm = float(request.form['maxtempm'])
        mintempm = float(request.form['mintempm'])
        maxhumidity_1 = float(request.form['maxhumidity_1'])
        minhumidity_1 = float(request.form['minhumidity_1'])
        maxdewptm_1 = float(request.form['maxdewptm_1'])
        mindewptm_1 = float(request.form['mindewptm_1'])
        maxpressurem_1 = float(request.form['maxpressurem_1'])
        minpressurem_1 = float(request.form['minpressurem_1'])
        
        # Call the function to make predictions using these input values
        dewpt_prediction = dewpt([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1])
        temp_prediction = temp([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1])
        r_fall = rainfall([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1])
        # Render the template with the predictions
        print(r_fall)
        return render_template("predict.html", dewpt_prediction=round(dewpt_prediction[0],2), temp_prediction=round(temp_prediction[0],2),rainfall=r_fall[0])
    else:
        return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)