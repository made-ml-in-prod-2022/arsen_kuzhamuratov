# Homework 1

 heart-disease-cleveland-uci dataset: https://www.kaggle.com/datasets/cherngs 

 ## Run training code
 
Gradient Boosting Classifier:

 `python ml_project/main.py --config-name grad_boosting files.data_path=path/to/training/data.csv`

  Logistic Regression Classifier:

`python ml_project/main.py --config-name log_reg files.data_path=path/to/training/data.csv`

## Run testing code

`python ml_project/main.py --config-name log_reg inference=True files.data_path=path/to/training/data.csv`

default data: `./ml_project/data/heart_cleveland_upload.csv`

inference data can be with/without labels: column "condition"

## Project Organization
```
├── LICENSE
├── ml_project
│   ├── config.py
│   ├── configs
│   │   ├── grad_boosting.yaml
│   │   └── log_reg.yaml
│   ├── data
│   │   ├── heart_cleveland_upload.csv
│   │   └── testing.csv
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
├── notebooks
│   └── EDA_analysis.ipynb
├── outputs
│   ├── column_mean_std_stats.sav
│   ├── file.log
│   ├── gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   └── prediction.csv
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── test_main.py
    └── test_utils.py
```