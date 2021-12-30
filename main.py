import numpy as np
from flask import Flask, request, render_template
import pickle
import random
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("models/random_forest2_SMOTE.pkl", "rb"))

# Data retrieved from the notebook thanks to describe function
features_config = {
    'Attr12': {'label': 'Gross Profit / Short-Term Liabilities', 'min': -6331.8, 'max': 8259.4},
    'Attr13': {'label': '(Gross Profit + Depreciation) / Sales', 'min': -1460.6, 'max': 13315.0},
    'Attr15': {'label': '(Total Liabilities * 365) / (Gross Profit + Depreciation)', 'min': -9632400.0,
               'max': 10236000.0},
    'Attr21': {'label': 'sales (n) / sales (n-1)', 'min': -1325.0, 'max': 29907.0},
    'Attr24': {'label': 'Gross Profit (In 3 Years) / Total Assets', 'min': -463.89, 'max': 831.66},
    'Attr25': {'label': '(Equity - Share Capital) / Total Assets', 'min': -500.93, 'max': 890.24},
    'Attr27': {'label': 'Profit On Operating Activities / Financial Expenses', 'min': -259010.0, 'max': 4208800.0},
    'Attr29': {'label': 'Logarithm Of Total Assets', 'min': -0.88606, 'max': 9.6983},
    'Attr30': {'label': '(Total Liabilities - Cash) / Sales', 'min': -6351.7, 'max': 152860.0},
    'Attr34': {'label': 'Operating Expenses / Total Liabilities', 'min': -1696.0, 'max': 21944.0},
    'Attr36': {'label': 'Total Sales / Total Assets', 'min': -0.00085719, 'max': 9742.3},
    'Attr37': {'label': '(Current Assets - Inventories) / Long-Term Liabilities', 'min': -525.52, 'max': 398920.0},
    'Attr4': {'label': 'Current Assets / Short-Term Liabilities', 'min': -0.40311, 'max': 53433.0},
    'Attr40': {'label': '(Current Assets - Inventory - Receivables) / Short-Term Liabilities', 'min': -101.27,
               'max': 8007.1},
    'Attr41': {'label': 'Total Liabilities / ((Profit On Operating Activities + Depreciation) * (12/365))',
               'min': -1234.4, 'max': 288770.0},
    'Attr43': {'label': 'Rotation Receivables + Inventory Turnover In Days', 'min': -115870.0, 'max': 30393000.0},
    'Attr45': {'label': 'Net Profit / Inventory', 'min': -256230.0, 'max': 366030.0},
    'Attr46': {'label': '(Current Assets - Inventory) / Short-Term Liabilities', 'min': -101.26, 'max': 53433.0},
    'Attr5': {'label': '[(Cash + Short-Term Securities + Receivables - Short-Term Liabilities) / (Operating Expenses '
                       '- Depreciation)] * 365', 'min': -11903000.0, 'max': 1250100.0},
    'Attr50': {'label': 'Current Assets / Total Liabilities', 'min': -0.045239, 'max': 53433.0},
    'Attr55': {'label': 'Working Capital', 'min': -1805200.0, 'max': 6123700.0},
    'Attr57': {'label': '(Current Assets - Inventory - Short-Term Liabilities) / (Sales - Gross Profit - Depreciation)',
               'min': -1667.3, 'max': 552.64},
    'Attr59': {'label': 'Long-Term Liabilities / Equity', 'min': -327.97, 'max': 23853.0},
    'Attr6': {'label': 'Retained Earnings / Total Assets', 'min': -508.41, 'max': 543.25},
    'Attr64': {'label': 'Sales / Fixed Assets', 'min': -10677.0, 'max': 294770.0},
    'Attr9': {'label': 'Sales / Total Assets', 'min': -3.496, 'max': 9742.3}
}


@flask_app.route("/")
def home():
    return render_template("index.html", features_config=features_config)


@flask_app.route("/predict", methods=["POST"])
def predict():
    features = pd.DataFrame(data=request.form, index=[0])
    features_dict = {}
    action = "predict"

    for feature, value in features.to_dict().items():
        if not value[0]:
            action = "fill"
            features_dict[feature] = round(
                random.uniform(features_config[feature]['min'], features_config[feature]['max']), 2)
        else:
            features_dict[feature] = float(value[0])

    if action == "predict":
        prediction = model.predict(pd.DataFrame(data=features_dict, index=[0]))
    else:
        prediction = False

    if isinstance(prediction, bool) and not prediction:
        prediction_text = (
            "Some values were empty, they have been filled randomly (between the minimum and maximum value of the "
            "feature in the dataset), check them and click again on predict to see the result. "
        )
        prediction_class = "warning"

    elif prediction == 0:
        prediction_text = "The Company will not bankrupt"
        prediction_class = "success"

    else:
        prediction_text = "The Company will bankrupt"
        prediction_class = "danger"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        prediction_class=prediction_class,
        features_dict=features_dict,
        features_config=features_config
    )


if __name__ == "__main__":
    flask_app.run()
