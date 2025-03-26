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

# ✅ Prediction Route (API Endpoint)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert input to DataFrame
        df = pd.DataFrame(data, index=[0])

        # Make prediction
        prediction = model.predict(df)

        # Return prediction
        return jsonify({"customer_satisfaction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
