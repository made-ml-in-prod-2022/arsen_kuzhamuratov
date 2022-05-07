# Homework 1

 Использовался стандартный датасет: https://www.kaggle.com/datasets/cherngs heart-disease-cleveland-uci

 ## Run training code
 
Gradient Boosting Classifier:

 `python ml_project/main.py configs/gradient_boosting.yaml --train -d path/to/training/data.csv`

  Logistic Regression Classifier:

`python ml_project/main.py configs/logistic_regression.yaml --train -d path/to/training/data.csv`

## Run testing code

`python ml_project/main.py configs/logistic_regression.yaml --inference -d path/to/training/data.csv`

training data: `./ml_project/data/heart_cleveland_upload.csv`

#to do testing data