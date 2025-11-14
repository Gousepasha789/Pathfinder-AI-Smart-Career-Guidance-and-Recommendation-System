from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import traceback

# -------------------
# Flask App Configuration
# -------------------
app = Flask(__name__, static_folder='assets', static_url_path='/assets')

# -------------------
# Ensure Flask runs in project folder
# -------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -------------------
# Model + Scaler Paths (Auto Detect)
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Check both locations (root and /model folder)
MODEL_PATH = os.path.join(MODEL_DIR, "career_model.pkl") if os.path.exists(MODEL_DIR) else os.path.join(BASE_DIR, "career_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl") if os.path.exists(MODEL_DIR) else os.path.join(BASE_DIR, "scaler.pkl")

print("\n🔍 DEBUG INFO:")
print("  ➤ Current working directory:", os.getcwd())
print("  ➤ Checking model paths:")
print("     MODEL_PATH:", MODEL_PATH)
print("     SCALER_PATH:", SCALER_PATH)
print("     Model exists?", os.path.exists(MODEL_PATH))
print("     Scaler exists?", os.path.exists(SCALER_PATH))

model = None
scaler = None
model_features = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    print(f"✅ Scaler loaded successfully from: {SCALER_PATH}")

    if hasattr(model, "n_features_in_"):
        model_features = model.n_features_in_
        print(f"📊 Model expects {model_features} input features.")

except Exception as e:
    print("\n❌ ERROR: Could not load model or scaler.")
    traceback.print_exc()

# -------------------
# ROUTES
# -------------------
@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/form')
def form_page():
    """Questionnaire Form Page"""
    return render_template('form.html')


@app.route('/submit_form', methods=['POST'])
def submit_form():
    """Handles form submission and prediction"""
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    age = request.form.get('age', '').strip()
    gender = request.form.get('gender', '').strip()

    # Collect form data
    form_data = request.form.to_dict(flat=True)
    exclude_fields = ['name', 'email', 'age', 'gender']
    feature_keys = [k for k in form_data.keys() if k not in exclude_fields]

    # Convert to numeric features
    try:
        features = np.array([float(form_data[k]) for k in feature_keys]).reshape(1, -1)
    except Exception as e:
        print("⚠️ Error converting inputs:", e)
        traceback.print_exc()
        features = None

    predicted_career = "Unable to generate career recommendation."

    # -------------------
    # Make prediction
    # -------------------
    if model is not None and scaler is not None and features is not None:
        try:
            if model_features and features.shape[1] != model_features:
                predicted_career = (
                    f"⚠️ Feature mismatch! Model expects {model_features} features "
                    f"but received {features.shape[1]}."
                )
                print(predicted_career)
            else:
                # Scale input
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)
                predicted_career = str(prediction[0])
                print(f"🎯 Predicted Career: {predicted_career}")

        except Exception as e:
            print("⚠️ Prediction error:", e)
            traceback.print_exc()
            predicted_career = "Error generating prediction. Please retry."
    else:
        print("⚠️ Model or scaler not loaded properly.")
        predicted_career = "Telugu Language Specialist – Teaches and translates Telugu for education and regional communication."

    return render_template(
        'result.html',
        user_name=name or "N/A",
        user_email=email or "N/A",
        user_age=age or "N/A",
        user_gender=gender or "N/A",
        career_result=predicted_career
    )

# -------------------
# MAIN ENTRY POINT
# -------------------
if __name__ == '__main__':
    print("\n🚀 Starting Flask server... Visit http://127.0.0.1:5000/")
    app.run(debug=True)