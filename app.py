from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        prediction = classifier.predict(data)
        
        if prediction == 1:
            return render_template('insulin_dosage.html', prediction_text="Oops! You have DIABETES.")
        else:
            return render_template('no_diabetes.html', prediction_text="Wow! You DON'T have diabetes.")

@app.route('/predict_insulin_dosage', methods=['POST'])
def predict_insulin_dosage():
    # Get the input features for insulin dosage prediction
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    bp = int(request.form['bloodpressure'])
    
    # Calculate insulin dosage (example calculation, adjust as needed)
    insulin_dosage = (insulin + bmi + dpf + bp) * 0.1
    
    return render_template('insulin_dosage_result.html', insulin_dosage=insulin_dosage)

if __name__ == '__main__':
    app.run(debug=True)
