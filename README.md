# Homework 1

 heart-disease-cleveland-uci dataset: https://www.kaggle.com/datasets/cherngs 

 ## Run training code
 
Gradient Boosting Classifier:

 `python ml_project/main.py configs/gradient_boosting.yaml --train -d path/to/training/data.csv`

  Logistic Regression Classifier:

`python ml_project/main.py configs/logistic_regression.yaml --train -d path/to/training/data.csv`

## Run testing code

`python ml_project/main.py configs/logistic_regression.yaml --inference -d path/to/inference/data.csv`

training data: `./ml_project/data/heart_cleveland_upload.csv`

inference data can be with/without labels: column "condition"

## Project Organization
├── configs\
│   ├── gradient_boosting.yaml\
│   └── logistic_regression.yaml\
├── LICENSE\
├── ml_project\
│   ├── data\
│   │   ├── heart_cleveland_upload.csv\
│   │   └── testing.csv\
│   ├── __init__.py\
│   ├── main.py\
│   └── utils.py\
├── notebooks\
│   └── EDA_analysis.ipynb\
├── outputs\
│   ├── column_mean_std_stats.sav\
│   ├── file.log\
│   ├── gradient_boosting.pkl\
│   ├── logistic_regression.pkl\
│   └── prediction.csv\
├── README.md\
├── requirements.txt\
├── setup.py\
└── tests\
    ├── test_main.py\
    └── test_utils.py
