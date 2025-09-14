# Maternal Health Risk
This project explores the Maternal Health dataset and develops machine learning models to predict maternal risk levels. It demonstrates the full **data science workflow**: from exploratory data analysis to model training and hyperparameter tuning, pipeline automation, and deployment via a Streamlit app. 

## Dataset
* Source: [UCI Machine Learning Repository]
* Size: < 1000 records, 6 health parameters
* Features: Age, Systolic BP, Diastolic BP, Blood Sugar, Body Temperature, Heart Rate
* Target: Maternal Risk (Low/Mid/High)

## [Exploratory Data Analysis](notebook/EDA.ipynb) (EDA)
* Univariate analysis: distributions of target and each health parameter
* Bivariate analysis: relationships between features and maternal risk
* Multivariate analysis: correlation heatmaps for identifying redundant features and creating dataset of smaller feature space
* Outlier detection and removal

## [Model Training & Tuning](notebook/Models.ipynb)
* Classifiers trained:
  * Softmax Regression
  * KNN (K-Nearest Neighbors)
  * Naive Bayes
  * Decision Tree
  * Random Forest
  * AdaBoost
  * Gradient Boosting
  * SVM (Support Vector Machine)
  * XGBoost (eXtreme Gradient Boosting)
  * CatBoost (Categorical Boosting)
* Hyperparameter tuning via GridSearchCV
* Evaluation on test set
* Results:
  * Best accuracy ~ 84%
  * Model: XGBoost
  * Models trained on the full feature space consistently outperformed those trained on the reduced feature set.

## [End-to-End ML Pipeline](src)
* Modular architecture with components for:
  * Data Ingestion
  * Data Transformation
  * Model Training (& hyperparameter tuning)
 * Train pipeline that integrates and runs all the components and saves the best model.
 * Predict pipeline which makes prediction for new input data.
 * Includes custom logging and exception handling.

## Deployment ([Streamlit App](https://maternal-health-risk-predictor.streamlit.app/))
* Interactive web app for real-time predictions.
* Handles user input of patient health parameters, executes the predict pipeline, and displays the predicted maternal risk level.

## Limitations
* Since the dataset is small, unrealistic values in data are not handled well by the model.
* Testing random input values on the app demonstrated that the model relies heavily on blood sugar and systolic bp (although relatively lesser than the former), while somewhat ignoring values (even humanly unrealistic ones) for other parameters.
* Hence, the predictions are not medically reliable - this project is for educational and demonstration purposes only.
