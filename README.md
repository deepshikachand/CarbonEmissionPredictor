# Carbon Emission Predictor

A web application that predicts your annual carbon emissions (in tons/year) based on your lifestyle and consumption habits. Powered by a machine learning model and built with Flask.

## Features
- Modern, responsive UI for easy data entry
- Predicts annual carbon emissions based on user input
- Instant results with AJAX (no page reload)
- Model, scaler, and feature columns loaded from pre-trained files

## Getting Started

### Requirements
- Python 3.7+
- Flask
- joblib
- pandas
- numpy
- xgboost

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepshikachand/CarbonEmissionPredictor

   ```
2. **Install dependencies:**
   ```bash
   pip install flask joblib pandas numpy
   ```
3. **Ensure the following files are present in the project root:**
   - `model.pkl` (trained ML model)
   - `scaler.pkl` (scaler used during training)
   - `columns.pkl` (list of feature columns)

### Running the App
```bash
python app.py
```
Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## Usage
1. Fill in the form with your monthly/weekly habits and lifestyle choices.
2. Click **Predict Emission**.
3. Your predicted annual carbon emission will appear instantly below the form.

## File Structure
```
CarbonFootPrint/
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── templates/
│   └── index.html
└── README.md
```

## Credits
- UI and backend: [Your Name]
- Machine learning model: [Your Data/Source]

## License
This project is for educational use only and is not licensed for commercial use.
