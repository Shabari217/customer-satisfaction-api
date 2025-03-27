from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("customer_satisfaction_model.pkl")

# ✅ Define Encoding for Ticket Type (Ensure it Matches Training)
ticket_type_mapping = {
    "Technical issue": 0,
    "Billing inquiry": 1,
    "Account support": 2,
    "General inquiry": 3,
    "Delivery Issue": 4  # Adjust based on training data
}

# ✅ Home Route (Renders HTML Page)
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Features Endpoint (Returns Expected Features)
@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({"expected_features": ["Customer Age", "Ticket Priority", "Ticket Type"]})

# ✅ Prediction Route (API Endpoint)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Validate that required features exist
        required_features = ["Customer Age", "Ticket Priority", "Ticket Type"]
        missing_features = [feat for feat in required_features if feat not in data]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # ✅ Encode "Ticket Type" to a number
        if df["Ticket Type"][0] not in ticket_type_mapping:
            return jsonify({"error": f"Invalid Ticket Type: {df['Ticket Type'][0]}. Allowed values: {list(ticket_type_mapping.keys())}"}), 400
        
        df["Ticket Type"] = df["Ticket Type"].map(ticket_type_mapping)

        # Make prediction
        prediction = model.predict(df)

        # Return prediction
        return jsonify({"customer_satisfaction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
