from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Check if model files exist
model_files = ["model.pkl", "scaler.pkl", "columns.pkl"]
missing_files = [f for f in model_files if not os.path.exists(f)]

if missing_files:
    print(f"WARNING: Missing model files: {missing_files}")
    model = None
    scaler = None
    columns = None
else:
    # Load model, scaler, and columns used during training
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")  # Trained feature list
        print("Model files loaded successfully")
    except Exception as e:
        print(f"Error loading model files: {e}")
        model = None
        scaler = None
        columns = None

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
    # Check if model is loaded
    if model is None or scaler is None or columns is None:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error="Model not loaded. Please check server logs."), 500
        return render_template('index.html', error="Model not loaded. Please check server logs.")
    
    try:
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

        # Handle Sex (radio buttons)
        sex_val = form.get("Sex")
        if sex_val == "male":
            input_dict["Sex_male"] = 1
        else:
            input_dict["Sex_male"] = 0  # female

        # Handle Body Type (radio buttons)
        body_type_val = form.get("Body_Type")
        if body_type_val and body_type_val != "normal":
            body_type_col = f"Body Type_{body_type_val}"
            if body_type_col in columns:
                input_dict[body_type_col] = 1

        # Handle transport (radio input)
        transport_val = form.get("Transport")
        if transport_val and transport_val != "private":
            transport_col = f"Transport_{transport_val}"
            if transport_col in columns:
                input_dict[transport_col] = 1

        # Handle Diet (radio input)
        diet_val = form.get("Diet")
        if diet_val and diet_val != "omnivore":
            diet_col = f"Diet_{diet_val}"
            if diet_col in columns:
                input_dict[diet_col] = 1

        # Handle How Often Shower (radio input)
        shower_val = form.get("How_Often_Shower")
        if shower_val and shower_val != "daily":
            shower_col = f"How Often Shower_{shower_val}"
            if shower_col in columns:
                input_dict[shower_col] = 1

        # Handle Social Activity (radio input)
        social_val = form.get("Social_Activity")
        if social_val and social_val != "never":
            social_col = f"Social Activity_{social_val}"
            if social_col in columns:
                input_dict[social_col] = 1

        # Handle Air Travel (radio input)
        air_val = form.get("Frequency_of_Traveling_by_Air")
        if air_val and air_val != "very_frequently":
            air_col = f"Frequency of Traveling by Air_{air_val}"
            if air_col in columns:
                input_dict[air_col] = 1

        # Handle Waste Bag Size (radio input)
        waste_val = form.get("Waste_Bag_Size")
        if waste_val and waste_val != "extra_large":
            waste_col = f"Waste Bag Size_{waste_val}"
            if waste_col in columns:
                input_dict[waste_col] = 1

        # Handle Energy Efficiency (radio input)
        energy_val = form.get("Energy_efficiency")
        if energy_val and energy_val != "no":
            energy_col = f"Energy efficiency_{energy_val}"
            if energy_col in columns:
                input_dict[energy_col] = 1

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

        # Convert from kg/year to tons/year (1 ton = 1000 kg)
        prediction_tons = prediction / 1000

        # AJAX response or form submit result
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(result=round(float(prediction_tons), 2))
        return render_template('index.html', result=round(float(prediction_tons), 2))
        
    except Exception as e:
        print(f"Prediction error: {e}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error=f"Prediction failed: {str(e)}"), 500
        return render_template('index.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
