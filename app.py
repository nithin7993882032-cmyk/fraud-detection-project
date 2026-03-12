from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# create flask app
app = Flask(__name__)

# load dataset
data = pd.read_csv("creditcard.csv")

# split features and label
X = data.drop("Class", axis=1)
y = data["Class"]

# train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        amount = float(request.form["amount"])

        # take sample row and change amount
        sample = X.iloc[0].copy()
        sample["Amount"] = amount

        prediction = model.predict([sample])

        if prediction[0] == 1:
            result = "⚠ Fraud Transaction Detected"
        else:
            result = "✅ Normal Transaction"

    return render_template("index.html", result=result)


# IMPORTANT for deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
