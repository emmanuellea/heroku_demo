from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page', methods=['POST'])
def page():

    dataset=pd.read_csv('Predict_BMI.csv')
    features=dataset[['Height M', 'Weight kg', '%Fat']]
    label=dataset['BMI']

    x_train, x_test, y_train, y_test=train_test_split(features, label, test_size=0.25, random_state=0)

    model=LinearRegression()
    model.fit(x_train, y_train)
    model.predict(x_test)
    

    '''
    sample=[[223,54,76]]
    predicted=model.predict(sample)[0]
    return render_template('page.html', result=predicted)

    '''
    if request.method=='POST':
        height=(int)(request.form['height'])
        weight=(int)(request.form['weight'])
        fat=(int)(request.form['fat'])
        input_variables = [[height, weight, fat]]
        predicted=model.predict(input_variables)[0]
        return render_template('page.html', result=round(predicted,3))

    
app.run(debug=True)
    


    
    
    

        
    
           
