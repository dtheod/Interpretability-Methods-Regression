import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor



class DataFrameImputer(TransformerMixin):

    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
        if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
    index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def imputting_or_removing_or_deleting(csv, threshold):
    initial_shape_row, initial_shape_col = csv.shape

    for cols in csv.columns:
        tmp_est = np.divide(csv[str(cols)].count(), initial_shape_row)
        if tmp_est <= threshold:
            csv = csv.drop([str(cols)], axis = 1)
    
    csv = DataFrameImputer().fit_transform(csv)
    return(csv)

def remove_nas(after_imputation):
    after_imputation['ct_nas'] = after_imputation.isnull().sum(axis=1)
    mean = np.mean(after_imputation['ct_nas'], axis = 0)
    standard_deviation = np.std(after_imputation['ct_nas'], axis=0)
    print(mean + 2 * standard_deviation)
    after_imputation = after_imputation.loc[after_imputation['ct_nas'] < mean + 2 * standard_deviation]
    return after_imputation

def feature_engineering(features_init):
    
    #average sales per neighborhood
    def typical_aggregations(feat_ini):
        aggregation = feat_ini.groupby('Neighborhood', as_index=False).agg({"SalePrice": "mean"})
        aggregation.columns = ['Neighborhood', 'Sales_by_Neighborhood']
        feat_ini = pd.merge(feat_ini,aggregation, how='left', on= 'Neighborhood')
        feat_ini['YrSold'] = feat_ini['YrSold'].astype('category')
        aggregation1 = feat_ini.groupby('YrSold', as_index=False).agg({"SalePrice": "mean"})
        aggregation1.columns = ['YrSold', 'Sales_by_Year']
        feat_ini = pd.merge(feat_ini,aggregation1, how='left', on= 'YrSold')
        return(feat_ini)
        
    return typical_aggregations(features_init)
 
def one_hot_encoding(after_imput):
    #Change dtypes into categories
    all_data_types = list(after_imput.dtypes)
    columns = list(after_imput)
    for data_type, colum in zip(all_data_types, columns):
        if str(data_type) == 'object':
            after_imput[colum] = after_imput[colum].astype('category')
    return pd.get_dummies(after_imput)
        

def train_test_split_func(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y,
    test_size = 0.3,random_state = 1)
    return X_train, y_train, X_test, y_test


def machine_learning_models(one_hot_ready):
    
    X = one_hot_ready.drop('SalePrice', axis=1).values
    Y = one_hot_ready['SalePrice'].values
    
    def scaling_data(X):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        return(scaler.fit_transform(X))
    
    """Random Forest Regression"""
    
    forest = RandomForestRegressor(n_estimators = 2000,
                                criterion = 'mse',
                                random_state = 1,
                                n_jobs = 200,
                                verbose = False)

    """Bayesian Ridge Regression"""
        
    clf = linear_model.BayesianRidge(n_iter = 1000,tol = 0.00001)
    X_scaled = scaling_data(X) 
    clf.fit(X_scaled,Y)
    predictions_net = clf.predict(X_scaled)

    predictions_net = predictions_net.reshape(-1,1)
    X = np.concatenate((X_scaled, predictions_net), axis = 1)
    X_scaled = scaling_data(X)

    """ K FOLD CROSS VALIDATION """
    
    kf = KFold(n_splits=5)
    n_ks = 0
    rmse_array = []

    for train_index, test_index in kf.split(X_scaled):
        n_ks += 1
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        forest.fit(X_train,y_train)
        predictions_rf = forest.predict(X_test)
        print("{} Fold has RMSE equal to {}".
              format(n_ks, validation_rmse(predictions_rf, y_test)))
        rmse_array.append(validation_rmse(predictions_rf, y_test))
    print("The Average Cross Validation Error is {}".
          format(round(sum(rmse_array)/len(rmse_array),)))
    return None


def validation_rmse(predicted, actual):
    return np.sqrt(mean_squared_error(np.log(actual), np.log(predicted)))

def actual_values_validation(predicted, actual):
    predicted_data = pd.DataFrame(predicted)
    actual_data = pd.DataFrame(actual)
    pred_and_actual = pd.concat([predicted_data, actual_data], axis=1)
    pred_and_actual.columns = ['predicted', 'actual']
    return  pred_and_actual


def master_function():
    
    #read real estate data
    train = pd.read_csv("Real_Estate_train.csv", index_col=0)
    #test = pd.read_csv("Real_Estate_test.csv", index_col=0)
    print(train.shape)
    #Cleaning data
    after_na_removal = remove_nas(train)
    after_imputation = imputting_or_removing_or_deleting(after_na_removal, 0.8)
    features_ready = feature_engineering(after_imputation)
    one_hot_ready = one_hot_encoding(features_ready)
    machine_learning_models(one_hot_ready)
    return None


if __name__ == "__main__":
    master_function()