import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dummy dataset (Replace this with actual dataset)
data = {
    "Customer Age": [25, 30, 45, 50, 60],
    "Ticket Type": [1, 2, 1, 3, 2],
    "Ticket Priority": [3, 1, 2, 3, 1],
    "Customer Satisfaction Rating": [5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)

# Define features and target
X = df.drop(["Customer Satisfaction Rating"], axis=1)
y = df["Customer Satisfaction Rating"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "customer_satisfaction_model.pkl")

print("✅ Model retrained and saved as 'customer_satisfaction_model.pkl'")
