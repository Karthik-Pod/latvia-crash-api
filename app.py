from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)
model = joblib.load('rf_pipeline.pkl')

@app.route('/', methods=['GET'])
def health():
    return "API is running", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    df_raw = pd.DataFrame([{
        'Age':         data['Age'],
        'Hour_sin':    data['Hour_sin'],
        'Hour_cos':    data['Hour_cos'],
        'AlcoholFlag': data['AlcoholFlag'],
        'Month':       data['Month'],
        'Weekday':     data['Weekday'],
        'IsWeekend':   data['IsWeekend']
    }])
   
    pred = int(model.predict(df_raw)[0])
    proba = model.predict_proba(df_raw)[0].tolist()
    return jsonify(predicted_severity=pred, class_probabilities=proba)

    
    X = np.array([fv], dtype=float)
  
    pred_class = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0].tolist()
    
    return jsonify(
        predicted_severity=pred_class,
        class_probabilities=pred_proba
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
