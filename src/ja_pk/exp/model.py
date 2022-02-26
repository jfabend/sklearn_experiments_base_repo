from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

class Model():

    def __init__(self, model_name):
        self.model_name = model_name
    
    def return_model(self):
        if self.model_name == "linearmodel":
            model = linear_model.LinearRegression()
        if self.model_name == "randomforestclassifier":
            model = RandomForestClassifier()
        if self.model_name == "randomforestregressor":
            model = RandomForestRegressor()
        if self.model_name == "xgboostregressor":
            model = XGBRegressor(objective='reg:squarederror')
        if self.model_name == "logisticregression":
            model = linear_model.LogisticRegression()
        if self.model_name == "xgboostclassifier":
            model = XGBClassifier()
        if self.model_name == "catboostregressor":
            model = CatBoostRegressor()
        if self.model_name == "catboostclassifier":
            model = CatBoostClassifier()
        if self.model_name == "svm":
            model = svm.SVC()
        if self.model_name == "mlpclassifier":
            model = MLPClassifier()
        if self.model_name == "lgbmclassifier":
            model = LGBMClassifier(num_leaves=24,
                                    min_child_samples=350,
                                    max_depth=5,
                                    n_estimators=170)
        if self.model_name == "naivebayesclassifier":
            model = GaussianNB()
        if self.model_name == "adaboostclassifier":
            model = AdaBoostClassifier()
        return model