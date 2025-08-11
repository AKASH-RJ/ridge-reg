from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv("ridge_regression.csv")

# Features and target
X = df[['Advertising', 'Price', 'Season', 'CompetitorPrice']]
y = df['Demand']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# Save scaler and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'ridge_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        advertising = float(request.form['advertising'])
        price = float(request.form['price'])
        season = int(request.form['season'])
        competitor_price = float(request.form['competitor_price'])

        scaler = joblib.load('scaler.pkl')
        model = joblib.load('ridge_model.pkl')

        features_scaled = scaler.transform([[advertising, price, season, competitor_price]])
        prediction = model.predict(features_scaled)[0]

        return render_template('index.html', prediction_text=f"Predicted Demand: {prediction:.2f} units")
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
