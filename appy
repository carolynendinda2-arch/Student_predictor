from flask import Flask
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

@app.route('/')
def home():

    # Data (Study hours vs scores)
    hours = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    scores = np.array([20, 40, 50, 60, 80, 90])

    # Model
    model = LinearRegression()
    model.fit(hours, scores)

    # Prediction
    prediction = model.predict([[7]])

    # Plot
    plt.figure()
    plt.scatter(hours, scores)
    plt.plot(hours, model.predict(hours))
    plt.xlabel("Study Hours")
    plt.ylabel("Scores")

    # Save graph
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/graph.png")
    plt.close()

    return f"""
    <h1>📊 Student Score Prediction</h1>

    <p>Predicted score for 7 hours:</p>

    <h2>{prediction[0]:.2f}</h2>

    <img src="/static/graph.png" width="500">
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
