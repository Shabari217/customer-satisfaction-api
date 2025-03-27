from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load the trained model
try:
    model = joblib.load("customer_satisfaction_model.pkl")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# ✅ Define Encoding for Ticket Type (Ensure it Matches Training)
ticket_type_mapping = {
    "Technical issue": 0,
    "Billing inquiry": 1,
    "Account support": 2,
    "General inquiry": 3,
    "Delivery Issue": 4  # Adjust based on training data
}

# ✅ Define Expected Feature Order
expected_columns = ["Customer Age", "Ticket Type", "Ticket Priority"]

# ✅ Home Route (Renders HTML Page)
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Features Endpoint (Returns Expected Features)
@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({"expected_features": expected_columns})

# ✅ Prediction Route (API Endpoint)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get JSON input
        data = request.get_json()

        # ✅ Validate that required features exist
        missing_features = [feat for feat in expected_columns if feat not in data]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        # ✅ Convert input to DataFrame & maintain feature order
        df = pd.DataFrame([data])[expected_columns]

        # ✅ Encode "Ticket Type" (Handle invalid values)
        if df["Ticket Type"][0] not in ticket_type_mapping:
            return jsonify({
                "error": f"Invalid Ticket Type: {df['Ticket Type'][0]}. Allowed values: {list(ticket_type_mapping.keys())}"
            }), 400
        df["Ticket Type"] = df["Ticket Type"].map(ticket_type_mapping)

        # ✅ Ensure model is loaded before prediction
        if model is None:
            return jsonify({"error": "Model not loaded. Please check server logs."}), 500

        # ✅ Make prediction
        prediction = model.predict(df)

        # ✅ Return prediction as JSON
        return jsonify({
            "customer_satisfaction": int(prediction[0]),
            "message": "Prediction successful."
        })

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
