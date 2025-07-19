from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and expected column names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # List of 45 feature names used during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_dict = {}

    # Initialize all columns to 0
    for col in columns:
        input_dict[col] = 0

    # Parse form input
    form = request.form

    # Handle numeric inputs
    numeric_fields = [
        'Monthly_Grocery_Bill',
        'Vehicle_Monthly_Distance_Km',
        'Waste_Bag_Weekly_Count',
        'How_Long_TV_PC_Daily_Hour',
        'How_Many_New_Clothes_Monthly',
        'How_Long_Internet_Daily_Hour'
    ]

    for field in numeric_fields:
        val = form.get(field)
        if val:
            input_dict[field] = float(val)

    # Handle transport (radio buttons: value is either 'private', 'Transport_public', 'Transport_walk/bicycle')
    transport_val = form.get("Transport")
    if transport_val in ['Transport_public', 'Transport_walk/bicycle']:
        input_dict[transport_val] = 1
    # If 'private' or not selected, both remain 0 (base case)

    # Handle other one-hot fields and checkboxes
    for field in form:
        if field in columns and field not in numeric_fields and field != "Transport":
            input_dict[field] = 1

    # Convert to DataFrame and align with trained model's columns
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=columns, fill_value=0)

    # Scale input
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        result = float(prediction)
        return jsonify(result=round(result, 2))
    return render_template('index.html', result=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
