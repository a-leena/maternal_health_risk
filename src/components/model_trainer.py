import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_object_path = os.path.join("artifacts", "model.pkl")
    models = {
        "Softmax Regression": LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5, random_state=42), random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        # "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }
    params = {
        "Softmax Regression": {
            'C': [0.01, 0.05, 0.1, 0.5, 1, 10]
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        "Naive Bayes": {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "Decision Tree": {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'criterion': ['gini', 'entropy']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'bootstrap': [True, False]
        },
        "AdaBoost": {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
        },
        "Gradient Boosting": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        "SVM": {
            'C': [0.1, 0.5, 1, 5, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [0, 3, 5, 7, 10],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        # "LightGBM": {
        #     'n_estimators': [100, 200, 300, 500],
        #     'max_depth': [-1, 3, 5, 7],
        #     'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        #     'num_leaves': [30, 50, 100]
        # },
        "CatBoost": {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [2, 4, 6, 8, 10]
        }
    }

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        tuning_results = {}
        for model_name, model_params in self.model_trainer_config.params.items():
            print(f"\n======================================== {model_name} ========================================")
            model = self.model_trainer_config.models[model_name]
            grid = GridSearchCV(
                estimator = model,
                param_grid = model_params,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Best Params:", grid.best_params_)
            print("Best CV Score:", grid.best_score_)
            print("Test Accuracy of Best Model:", accuracy)
            tuning_results[model_name] = {
                'best_model': best_model,
                'best_params':grid.best_params_, 
                'best_test_accuracy': accuracy
            }
        highest_acc = float('-inf')
        best_model_name = None
        for name, tuning_res in tuning_results.items():
            if tuning_res['best_test_accuracy'] > highest_acc:
                highest_acc = tuning_res['best_test_accuracy']
                best_model_name = name
        return best_model_name, tuning_results[best_model_name]
    
    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Enter the model trainer component.")
        try:
            X_train, y_train, X_test, y_test = (train_array[:,:-1], 
                                                train_array[:,-1], 
                                                test_array[:, :-1], 
                                                test_array[:,-1])
            best_model_name, best_model_tuning_results = self.evaluate_model(X_train=X_train,
                                                                             y_train=y_train,
                                                                             X_test=X_test,
                                                                             y_test=y_test)
            save_object(
                file_path=self.model_trainer_config.trained_model_object_path,
                object=best_model_tuning_results['best_model']
            )
            logging.info("Best classification model found and saved.")
            return best_model_name, best_model_tuning_results['best_test_accuracy']

        except Exception as e:
            raise CustomException(e, sys)