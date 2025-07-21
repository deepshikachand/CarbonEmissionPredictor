# from flask import Flask, request, render_template, jsonify
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Load model, scaler, and expected column names
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")
# columns = joblib.load("columns.pkl")  # List of 45 feature names used during training

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     input_dict = {}

#     # Initialize all columns to 0
#     for col in columns:
#         input_dict[col] = 0

#     # Parse form input
#     form = request.form

#     # Handle numeric inputs
#     numeric_fields = [
#         'Monthly_Grocery_Bill',
#         'Vehicle_Monthly_Distance_Km',
#         'Waste_Bag_Weekly_Count',
#         'How_Long_TV_PC_Daily_Hour',
#         'How_Many_New_Clothes_Monthly',
#         'How_Long_Internet_Daily_Hour'
#     ]

#     for field in numeric_fields:
#         val = form.get(field)
#         if val:
#             input_dict[field] = float(val)

#     # Handle transport (radio buttons: value is either 'private', 'Transport_public', 'Transport_walk/bicycle')
#     transport_val = form.get("Transport")
#     if transport_val in ['Transport_public', 'Transport_walk/bicycle']:
#         input_dict[transport_val] = 1
#     # If 'private' or not selected, both remain 0 (base case)

#     # Handle other one-hot fields and checkboxes
#     for field in form:
#         if field in columns and field not in numeric_fields and field != "Transport":
#             input_dict[field] = 1

#     # Convert to DataFrame and align with trained model's columns
#     df = pd.DataFrame([input_dict])
#     df = df.reindex(columns=columns, fill_value=0)

#     # Scale input
#     scaled = scaler.transform(df)

#     # Predict
#     prediction = model.predict(scaled)[0]

#     if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#         result = float(prediction)
#         return jsonify(result=round(result, 2))
#     return render_template('index.html', result=round(prediction, 2))

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and columns used during training
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # Trained feature list

# Mapping of form field names to trained model's column names
field_map = {
    'Monthly_Grocery_Bill': 'Monthly Grocery Bill',
    'Vehicle_Monthly_Distance_Km': 'Vehicle Monthly Distance Km',
    'Waste_Bag_Weekly_Count': 'Waste Bag Weekly Count',
    'How_Long_TV_PC_Daily_Hour': 'How Long TV PC Daily Hour',
    'How_Many_New_Clothes_Monthly': 'How Many New Clothes Monthly',
    'How_Long_Internet_Daily_Hour': 'How Long Internet Daily Hour'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    input_dict = {}

    # Initialize all model columns with 0
    for col in columns:
        input_dict[col] = 0

    # Fill numeric fields after mapping them
    for html_field, model_field in field_map.items():
        val = form.get(html_field)
        if val:
            try:
                input_dict[model_field] = float(val)
            except ValueError:
                input_dict[model_field] = 0

    # Handle transport (radio input)
    transport_val = form.get("Transport")
    transport_col = f"Transport_{transport_val}"
    if transport_col in columns:
        input_dict[transport_col] = 1

    # Handle one-hot and checkbox inputs
    for field in form:
        # Skip numeric fields and 'Transport'
        if field in field_map or field == "Transport":
            continue
        # Some fields might match model columns directly
        if field in columns:
            input_dict[field] = 1
        else:
            # If the form field is something like "Sex_male", map it
            converted = field.replace("_", " ").lower()
            for col in columns:
                if col.lower() == converted:
                    input_dict[col] = 1
                    break

    # Create input DataFrame
    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]

    # AJAX response or form submit result
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(result=round(float(prediction), 2))
    return render_template('index.html', result=round(float(prediction), 2))

if __name__ == '__main__':
    app.run(debug=True)
