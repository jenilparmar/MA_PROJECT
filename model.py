import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask ,render_template,request
app = Flask(__name__)
def dewpt(dataOP):
    data = pd.read_csv("data\JaipurFinalCleanData.csv")
    data.dropna(inplace=True)
    #Check weather Any row's data missing or not if missing that fill it
    X = data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1',
              'maxpressurem_1','minpressurem_1']]
    # Features of dataset
    y = data['meandewptm_2']
    #Our target for Predict 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    # Splitting data into training and testing sets
    model = LinearRegression()
    # Initialize the linear regression model
    model.fit(X_train, y_train)
    #feed the training data
    y_pred = model.predict(X_test)
    # Model Evaluation
    # Predicting on the testing set
    new_data = pd.DataFrame([dataOP], columns=['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1'])
    # Prediction
    # Assuming new_data contains the features (Temperature, Humidity, Pressure, AirSpeed, RainToday) for a new day
    prediction = model.predict(new_data)
    # our predicted data :) 
    return prediction
def temp(dataOP):
    weather_data = pd.read_csv("data\JaipurFinalCleanData.csv")
    #Imporing Data
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
    X = weather_data[['maxtempm', 'mintempm', 'maxhumidity_1', 'minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1']]  
    y = weather_data['precipm_2_B_nextday']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Precipitation on test data : ",y_pred)

    new_data = pd.DataFrame([data], columns=['maxtempm', 'mintempm', 'maxhumidity_1','minhumidity_1','maxdewptm_1','mindewptm_1','maxpressurem_1','minpressurem_1','precipm_1'])

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
        precipm_1 = request.form['precipm_1']
        dewpt_prediction = dewpt([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1])
        temp_prediction = temp([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1])
        flag = False
        if precipm_1== '':
            r_fall = rainfall([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1,precipm_1])
            return render_template("predict.html", dewpt_prediction=round(dewpt_prediction[0],2), temp_prediction=round(temp_prediction[0],2),flag = flag,rainfall=r_fall[0])
            
        else:
            flag = True
            precipm_1 = float(precipm_1)
            r_fall = rainfall([maxtempm, mintempm, maxhumidity_1, minhumidity_1, maxdewptm_1, mindewptm_1, maxpressurem_1, minpressurem_1,precipm_1])
            return render_template("predict.html", dewpt_prediction=round(dewpt_prediction[0],2), temp_prediction=round(temp_prediction[0],2),flag = flag,rainfall=r_fall[0])
            
      
    else:
        return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)