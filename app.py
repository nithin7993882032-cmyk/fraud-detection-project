from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("creditcard.csv")

# Prepare data
X = data.drop("Class", axis=1)
y = data["Class"]

# Use smaller data for faster training
X_small = X.head(10000)
y_small = y.head(10000)

# Train model
model = RandomForestClassifier(n_estimators=20)
model.fit(X_small, y_small)

@app.route("/", methods=["GET","POST"])
def home():

    result = ""

    if request.method == "POST":

        amount = float(request.form["amount"])

        # create sample input (same length as dataset features)
        sample = [0]*29
        sample.append(amount)

        prediction = model.predict([sample])[0]

        if prediction == 1:
            result = "Fraud Transaction"
        else:
            result = "Normal Transaction"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)