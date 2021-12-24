import numpy as np
from flask import Flask, request, render_template
import pickle
import random
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("models/random_forest2_SMOTE.pkl", "rb"))

features_range = {
    'Attr12': {'min': -6331.8, 'max': 8259.4},
    'Attr13': {'min': -1460.6, 'max': 13315.0},
    'Attr15': {'min': -9632400.0, 'max': 10236000.0},
    'Attr21': {'min': -1325.0, 'max': 29907.0},
    'Attr24': {'min': -463.89, 'max': 831.66},
    'Attr25': {'min': -500.93, 'max': 890.24},
    'Attr27': {'min': -259010.0, 'max': 4208800.0},
    'Attr29': {'min': -0.88606, 'max': 9.6983},
    'Attr30': {'min': -6351.7, 'max': 152860.0},
    'Attr34': {'min': -1696.0, 'max': 21944.0},
    'Attr36': {'min': -0.00085719, 'max': 9742.3},
    'Attr37': {'min': -525.52, 'max': 398920.0},
    'Attr4': {'min': -0.40311, 'max': 53433.0},
    'Attr40': {'min': -101.27, 'max': 8007.1},
    'Attr41': {'min': -1234.4, 'max': 288770.0},
    'Attr43': {'min': -115870.0, 'max': 30393000.0},
    'Attr45': {'min': -256230.0, 'max': 366030.0},
    'Attr46': {'min': -101.26, 'max': 53433.0},
    'Attr5': {'min': -11903000.0, 'max': 1250100.0},
    'Attr50': {'min': -0.045239, 'max': 53433.0},
    'Attr55': {'min': -1805200.0, 'max': 6123700.0},
    'Attr57': {'min': -1667.3, 'max': 552.64},
    'Attr59': {'min': -327.97, 'max': 23853.0},
    'Attr6': {'min': -508.41, 'max': 543.25},
    'Attr64': {'min': -10677.0, 'max': 294770.0},
    'Attr9': {'min': -3.496, 'max': 9742.3}
}


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    features = pd.DataFrame(data=request.form, index=[0])
    features_dict = {}

    for feature, value in features.to_dict().items():
        if not value[0]:
            features_dict[feature] = float(random.uniform(features_range[feature]['min'], features_range[feature]['max']))
        else:
            features_dict[feature] = float(value[0])

    prediction = model.predict(pd.DataFrame(data=features_dict, index=[0]))
    if prediction == 0:
        return render_template("index.html",
                               prediction_text="The Company will not bankrupt  it's class is {}".format(prediction),
                               prediction_class="success")
    else:
        return render_template("index.html",
                               prediction_text=f"The Company will bankrupt it's class is {prediction}",
                               prediction_class="danger")


if __name__ == "__main__":
    flask_app.run(debug=True)
