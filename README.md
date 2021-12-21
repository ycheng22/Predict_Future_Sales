# Predict_Future_Sales

This is a Final project of course: How to Win a Data Science Competition: Learn from Top Kagglers.

The data is from: [Predict Future Sales, 
Final project for "How to win a data science competition" Coursera course](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)

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

`Part 1 Data Processing and Engineeringt.ipynb` includes:

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

`Part 2 Model Training and Prediction.ipynb` includes:

* [1. Model Training](#sec1)
    * [1.1 Validation scheme](#sec1_1)
    * [1.2 LightGBM Regressor](#sec1_2)
    * [1.3 XGBoost Regressor](#sec1_3)
        * [1.3.1 Training XGBoost Regressor on GPU](#sec1_3_1)
    * [1.4 Linear regression](#sec1_4)
* [2. Ensembling Method](#sec2)
    * [2.1 Ensembling: LightGBM + Linear Regression](#sec2_1)    
* [3. Predictions](#sec3)  
