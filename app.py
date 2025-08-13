from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("ridge_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            month = float(request.form["month"])
            price = float(request.form["price"])
            advertising = float(request.form["advertising"])
            competitor_price = float(request.form["competitor_price"])

            features = np.array([[month, price, advertising, competitor_price]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            prediction = round(prediction, 2)
        except:
            prediction = "Error in input values!"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
