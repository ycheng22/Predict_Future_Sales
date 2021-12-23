# Predict_Future_Sales

This is a Final project of course: How to Win a Data Science Competition: Learn from Top Kagglers.

The data is from: [Predict Future Sales, 
Final project for "How to win a data science competition" Coursera course](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)

**Table of Contents:**

- [Predict_Future_Sales](#predict_future_sales)
  - [`Part 1 Data Processing and Engineeringt.ipynb` includes:](#part-1-data-processing-and-engineeringtipynb-includes)
  - [`Part 2 Model Training and Prediction.ipynb` includes:](#part-2-model-training-and-predictionipynb-includes)
  - [Public leaderboard Score](#public-leaderboard-score)


**Folder Structure:**
 ```
 📦Predict_Future_Sales-main
 ┣ 📂models
 ┃ ┣ 📜model1_lgbm.pkl
 ┃ ┣ 📜model2_xgb.pkl
 ┃ ┣ 📜model3_linreg.pkl
 ┃ ┣ 📜model4_ridge.pkl
 ┃ ┗ 📜model5_lasso.pkl
 ┣ 📂submissions
 ┃ ┣ 📜submission.csv
 ┃ ┣ 📜submission_catboost.csv
 ┃ ┣ 📜submission_LASSO.csv
 ┃ ┣ 📜submission_LGBM.csv
 ┃ ┣ 📜submission_LINREG.csv
 ┃ ┣ 📜submission_RIDGE.csv
 ┃ ┗ 📜submission_XGBM.csv
 ┣ 📜Part 1 Data Processing and Engineering.ipynb
 ┣ 📜Part 1 Data Processing and Engineering.pdf
 ┣ 📜Part 2 Model Training and Prediction.ipynb
 ┣ 📜Part 2 Model Training and Prediction.pdf
 ┗ 📜README.md
 ```

## `Part 1 Data Processing and Engineeringt.ipynb` includes:

* [1. Importing data](#sec1)
* [2. Preprocessing data](#sec2)
    * [2.1 Extracting categorical features from Russian item names](#sec2_1)
    * [2.2 Creating text features with TF-IDF](#sec2_2)
    * [2.3 Check duplicates ](#sec2_3)
    * [2.4 Preprocessing for data engineering](#sec2_4)
    * [2.5 Create lagged features](#sec2_5)
    * [2.6 Creating time series trend features](#sec2_6)
    * [2.7 Feature matrix](#sec2_7)
* [3. Exploratory data analysis](#sec3)
    * [3.1 Target variable](#sec3_1)
    * [3.2 Multivariate heatmaps](#sec3_2)
* [4. Advancedd Feature Engineering](#sec4)
    * [4.1 Mean encoding on categorical features](#sec4_1)
    * [4.2 Matrix factorization of TFIDF processed features](#sec4_2)

**Comments:**
- Extracting categorical features such as city from `shop` names, extract item type/subtype from `item` names
- Creating text features from `shop` names, `item` names with `TF-IDF` method
- Creating lagged features, time series trend features for prediction
- Creating heatmap to discover the correlation
- Mean encoding on the categorical features, mean encoding is based on `the frequency of target variable appears on average in the categorical feature`
- Applying NMF to reduce dimensionality


## `Part 2 Model Training and Prediction.ipynb` includes:

* [1. Model Training](#sec1)
    * [1.1 Validation scheme](#sec1_1)
    * [1.2 LightGBM Regressor](#sec1_2)
    * [1.3 XGBoost Regressor](#sec1_3)
        * [1.3.1 Training XGBoost Regressor on GPU](#sec1_3_1)
    * [1.4 CatBoost Regressor on GPU](#sec1_4)
    * [1.5 Linear models](#sec1_5)
        * [1.5.1 LinearRegression model](#sec1_5_1)
        * [1.5.2 Ridge model](#sec1_5_2)
        * [1.5.3 Lasso model](#sec1_5_3)
* [2. Ensembling Method](#sec2)
    * [2.1 Ensembling: LightGBM + Linear Regression](#sec2_1)    
* [3. Predictions](#sec3) 

**Comments:**
- Trained model on Linear model, such as `LinearRegression, Ridge, Lasso`
- Trained model on Gradient boosing method, such as `LightGBM Regressor, XGBoost Regressor, CatBoost Regressor`
- Improving the speed of training by runnnig `XGBoost, CatBoost` on GPU
- For example, trained `XGBoost` on CPU cost 1 hour 23 minutes, while on GPU only cost 57 seconds, which are 3000* faster.
- Comparing the predictions of previous models with ensembling method

## Public leaderboard Score
| Model | CatBoost | XGBoost | LightGBM | LinearRegression | Ridge | Lasso  |
|-------|----------|---------|----------|------------------|-------|--------|
| Score | 0.96326  | 0.98971 | 0.98313  |      1.07821     |1.07822| 1.21744|



