from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("customer_satisfaction_model.pkl")

# ✅ Home Route (Renders HTML Page)
@app.route("/")
def home():
    return render_template("index.html")  # This will display your HTML page

# ✅ Features Route (Check Expected Columns)
@app.route("/features", methods=["GET"])
def get_features():
    try:
        feature_names = model.feature_names_in_.tolist()  # Get expected column names
        return jsonify({"expected_features": feature_names})
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Prediction Route (API Endpoint)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert input to DataFrame
        df = pd.DataFrame(data, index=[0])

        # ✅ Ensure only expected columns are used
        expected_features = model.feature_names_in_.tolist()

        # ✅ Check if the request contains all required features
        missing_features = [feature for feature in expected_features if feature not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"})

        # ✅ Ensure the DataFrame has only the expected columns
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)

        # Return prediction
        return jsonify({"customer_satisfaction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
