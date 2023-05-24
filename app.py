from flask import Flask, request, make_response, render_template
import numpy as np
import os
from PIL import Image
import joblib

app = Flask(__name__)

#prdct to imprt
l_reg=open("RegressionLS.pkl","rb")
ml_model=joblib.load(l_reg)

#set paths to upload folder
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
   if request.method == "POST" :
      try:
         YearsExperience = float(request.form["YearsExperience"])
         model_prediction = ml_model.predict(np.array([[YearsExperience]]))[0]
         model_prediction = round(float(model_prediction), 2)
         #model_prediction = ml_model.predict(np.array([[YearsExperience]]))[0]
        # model_prediction=round(float(model_prediction),2)
      except ValueError:
         return "please ckeck if the value are entered correctly"
   return render_template('predict.html',prediction=model_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0')