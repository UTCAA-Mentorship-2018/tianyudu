# Home Credit Default Risk

### What's This?

This repository contains source code for the home credit default risk prediction competition on Kaggle. (<https://www.kaggle.com/c/home-credit-default-risk>)

## Models and Methods

The central task of this project is a *binary classification problem*. In our basic model, only features from `application_train.csv`  are used.

Raw dataset contains 122 columns with around 300k observations in total.

Detailed statistics on the dataset can be explored in `./explore_data.ipynb`



### Data Pre-processing i: Dropping the Data

`ID` and `TARGET` are excluded from training features.

Many of columns involves invalid observations (marked as `nan` or empty). In our data preprocessing pipeline, we specify a threshold between 0% and 100% such that any feature involving more than the threshold set will be dropped.

We set the threshold to be 10% in our model. (i.e. any features containing more than 10% invalid observations will be excluded from our model.)



### Data Pre-processing ii: Encoding Data

### Data Pre-processing iii: Training, Testing and Validation sets.

### Model 1. Light Gradient Boosting Machine

