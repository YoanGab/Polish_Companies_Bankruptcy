# Python for Data Analysis

## Project - Polish Companies Bankruptcy Dataset

### Made by: Yoan Gabison


## Content

This repository contains:
- Dataset with the information about the bankruptcy of Polish companies.
- Jupyter Notebook for the analysis (Analysis + DataViz) of the Polish Companies Bankruptcy Dataset with ML models to predict if a company will make bankrupt or not.
- Python API with Flask to predict easily if a company will make bankrupt or not.
- PowerPoint Presentation of the project.
- The different models trained on the dataset.

## Description 

This project had for goal to analyse the assigned dataset.
My dataset was about bankruptcy prediction of Polish companies.
The bankrupt companies were analysed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013.

## Getting started

### Install the necessary libraries

To install the necessary libraries, run the following command in the terminal:

`pip install -r requirements.txt`


### Start the API

To start the API, run the following command in the terminal:

`python3 main.py`


## How to use the API

Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the API.


API when fields are not all filled, button is "Fill all fields"
![](https://github.com/YoanGab/Polish_Companies_Bankruptcy/blob/master/images/API_fields_empty.png)
<br>
API when all fields are filled, button becomes "Predict"
![](https://github.com/YoanGab/Polish_Companies_Bankruptcy/blob/master/images/API_fields_filled.png)
<br>
API predicts No Bankrupt for this company
![](https://github.com/YoanGab/Polish_Companies_Bankruptcy/blob/master/images/Success.png)
<br>
API predicts Bankrupt for this company
![](https://github.com/YoanGab/Polish_Companies_Bankruptcy/blob/master/images/Bankrupt.png)
