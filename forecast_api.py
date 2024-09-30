from flask import Flask, jsonify
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing

app = Flask(__name__)

@app.route('/forecast', methods=['GET'])
def forecast():
    # Load dataset
    series = AirPassengersDataset().load()

    # Set aside the last 36 months as a validation series
    train, val = series[:-36], series[-36:]

    # Initialize and train the model
    model = ExponentialSmoothing()
    model.fit(train)

    # Predict
    prediction = model.predict(len(val), num_samples=600)

    # Convert the prediction to a format that can be easily serialized to JSON
    forecast_dict = {
        "time": [str(time) for time in prediction.time_index],
        "forecast": prediction.values().tolist()
    }

    return jsonify(forecast_dict)

if __name__ == '__main__':
    app.run(debug=True)
