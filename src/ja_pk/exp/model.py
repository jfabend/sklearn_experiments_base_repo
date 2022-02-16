from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier

class Model():

    def __init__(self, model_name):
        self.model_name = model_name
    
    def return_model(self):
        if self.model_name == "linearmodel":
            model = linear_model.LinearRegression()
        if self.model_name == "randomforestclassifier":
            model = RandomForestClassifier()
        if self.model_name == "xgboostregressor":
            model = XGBRegressor(objective='reg:squarederror')
        if self.model_name == "logisticregression":
            model = linear_model.LogisticRegression()
        if self.model_name == "xgboostclassifier":
            model = XGBClassifier()
        if self.model_name == "svm":
            model = svm.SVC()
        if self.model_name == "mlpclassifier":
            model = MLPClassifier()
        return model