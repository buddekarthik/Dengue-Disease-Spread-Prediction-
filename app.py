import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/dd')  
def dd():
    return render_template('Disease Description.html')

@app.route('/maps')
def maps():
    return render_template('Maps.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    #a=float(int_features[-2])
    #int_features1 = int_features
    #int_features[0] = x_label.transform(np.array([int_features[0]]))
    #int_features = onehotencoder.transform([int_features]).toarray()
    #int_features = int_features[1:]
    #for i in int_features:
     #  int_features1.append(float(i)) 
    #final_features = int_features

    prediction = model.predict(np.array(int_features).reshape(1,-1))
    output = round(prediction[0], 2)    
    
    #output = round(prediction[0]*100000, 2)

    return render_template('index.html', prediction_text='Number of cases till then will be :{}'.format(output))

@app.route('/visual')
def visual():
    return render_template('visualize.html')

"""@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)"""

if __name__ == "__main__":
    app.run(debug=True)