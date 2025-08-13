---

#  Demand Forecasting using Ridge Regression

##  Project Overview

This project predicts **future demand** using the **Ridge Regression** algorithm, which is a type of linear regression with **L2 regularization** to prevent overfitting.
The model is deployed as a **Flask web application** with an HTML & CSS interface, allowing users to input relevant features and get forecasted demand instantly.

---

##  Technologies Used

* **Python 3.x**
* **Flask** – Web framework for deployment
* **Scikit-learn** – Machine learning library (Ridge Regression)
* **Pandas** – Data handling
* **NumPy** – Numerical computation
* **HTML & CSS** – Frontend interface

---

##  Project Structure

```
ridge_regression_demand_forecasting/
│── model.py           # Ridge Regression model training
│── app.py             # Flask application
│── templates/
│   └── index.html     # HTML form for user input
│── static/
│   └── style.css      # CSS styling
│── demand_dataset.csv # Dataset with 50 or 200 rows
│── README.md          # Project documentation
```

---

##  Dataset Description

The dataset contains historical demand data with several influencing factors.

**Example columns:**

* `month` – Month of observation
* `price` – Price of the product
* `advertising` – Advertising spend in that period
* `competitor_price` – Price of competitor’s product
* `demand` – Target variable (units sold)

---

##  Installation & Setup

1️ **Clone the repository**

```bash
git clone https://github.com/yourusername/ridge-regression-demand.git
cd ridge-regression-demand
```

2️ **Install dependencies**

```bash
pip install flask pandas numpy scikit-learn
```

3️ **Train the model**

```bash
python model.py
```

This will save the trained Ridge Regression model as `ridge_model.pkl`.

4️ **Run Flask app**

```bash
python app.py
```

Access the app in your browser at:

```
http://127.0.0.1:5000/
```

---

##  Usage

1. Open the app in your browser.
2. Enter the input values (price, advertising spend, etc.).
3. Click **Predict Demand**.
4. The model will output the forecasted demand.

---

##  Model Details

* **Algorithm**: Ridge Regression (L2 regularization)
* **Regularization Parameter (alpha)**: Tuned for best performance
* Helps avoid overfitting in cases where predictors are highly correlated

---

##  Screenshot
---   
Home Page
  <img width="620" height="486" alt="Screenshot 2025-08-13 110017" src="https://github.com/user-attachments/assets/e3aff99f-6220-4446-a86b-1f69b9b67a54" />

---

Predection Page

<img width="587" height="502" alt="Screenshot 2025-08-13 110031" src="https://github.com/user-attachments/assets/ba9a02ff-62a0-494e-8896-9e57819b450e" />
