# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# --- Load model safely ---
MODEL_PATH = r"C:\Users\sande\OneDrive\Desktop\A_VS_CODE\student mark predictor\student_mark_predictor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded model from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# in-memory dataframe to store entries (and saved to CSV)
df = pd.DataFrame(columns=["Study Hours", "Predicted Output"])
CSV_PATH = os.path.join(os.path.dirname(__file__), "smp_data_from_app.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df, model

    if model is None:
        # return a friendly message if model wasn't loaded
        return render_template('index.html', prediction_text='Model not loaded. Check server terminal for errors.')

    # read form values — assume first form field is study hours
    try:
        # allow decimals; convert to float
        input_values = list(request.form.values())
        if not input_values:
            return render_template('index.html', prediction_text='No input received.')

        study_hours = float(input_values[0])
    except ValueError:
        return render_template('index.html', prediction_text='Please enter a numeric value for study hours.')

    # validate hours (0-24)
    if study_hours < 0 or study_hours > 24:
        return render_template('index.html', prediction_text='Please enter valid hours between 0 and 24.')

    # prepare features for model: create 2D array (1, n_features)
    features = np.array([study_hours]).reshape(1, -1)

    # predict with safe handling
    try:
        pred = model.predict(features)  # usually returns array-like
        # extract scalar from returned structure
        if hasattr(pred, "__len__"):
            output = float(pred[0])
        else:
            output = float(pred)
        output = round(output, 2)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return render_template('index.html', prediction_text='Prediction failed. Check server terminal for details.')

    # append to dataframe and save CSV
    try:
        new_row = pd.DataFrame({"Study Hours": [study_hours], "Predicted Output": [output]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] Saved to {CSV_PATH}\n{df.tail()}")
    except Exception as e:
        print(f"[WARN] Failed to save CSV: {e}")

    return render_template(
        'index.html',
        prediction_text=f'You will get [{output}%] marks when you study [{study_hours}] hours per day.'
    )


if __name__ == "__main__":
    # For development only. Use a proper WSGI server in production.
    app.run(host='127.0.0.1', port=5000, debug=True)
