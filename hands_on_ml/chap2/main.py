import pandas as pd
import __init__
import src.load_data.loader as data_loader
import src.load_data.split as split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


class CombinedAttributesAdderHousing(BaseEstimator, TransformerMixin):
    def __init__(self, idxs, add_custom=True):
        self.add_custom = add_custom
        self.rooms_ix = idxs['rooms']
        self.bedrooms_ix = idxs['bedrooms']
        self.population_ix = idxs['population']
        self.household_ix = idxs['households']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.add_custom:
            rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
            return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def get_data():
    housing_path = data_loader.dataset_path("housing", "housing.csv")
    df = pd.read_csv(housing_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def print_map(data):
    data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/500, label='population',
              figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()


def GetHousingPipeline(housing):
    #housing_cat = pd.DataFrame(housing["ocean_proximity"])
    cat_attribs = ['ocean_proximity']
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attributes = list(housing_num)

    idxs = {'rooms': num_attributes.index('total_rooms'),
            'bedrooms': num_attributes.index('total_bedrooms'),
            'population': num_attributes.index('population'),
            'households': num_attributes.index('households')
            }
    print(idxs)

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdderHousing(idxs=idxs)),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat encoder', OneHotEncoder(sparse=False)),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    return full_pipeline


def CrossValidate(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    tree_reg = DecisionTreeRegressor()
    forest_reg = RandomForestRegressor()

    print("Cross Validating Linear Regression")
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print("Average RMSE", lin_rmse_scores.mean())

    print("Cross Validating Decision Tree")
    tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    print("Average RMSE", tree_rmse_scores.mean())

    print("Cross Validating Random Forest")
    forest_reg = RandomForestRegressor(n_estimators=5)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    print("Average RMSE", forest_rmse_scores.mean())


def train_and_save(path_name, housing_prepared, housing_labels):
    my_model = RandomForestRegressor(n_estimators=10)
    my_model.fit(housing_prepared, housing_labels)
    joblib.dump(my_model, path_name)


def test_saved_model(path_name, housing_prepared):
    my_model = joblib.load(path_name)
    predictions = my_model.predict(housing_prepared)
    rmse = mean_squared_error(housing_labels, predictions)
    print(np.sqrt(rmse))


def do_grid_search(housing_prepared, housing_labels):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    print("Feature Importances")
    np.set_printoptions(suppress=True)
    print(grid_search.best_estimator_.feature_importances_)
    return grid_search.best_estimator_

if __name__ == "__main__":
    train, test = get_data()
    housing_labels = train["median_house_value"].copy()
    housing = train.drop("median_house_value", axis=1)

    full_pipeline = GetHousingPipeline(housing)
    housing_prepared = full_pipeline.fit_transform(housing)
    #CrossValidate(housing_prepared, housing_labels)
    #train_and_save("my_model2.plk", housing_prepared=housing_prepared, housing_labels=housing_labels)
    #test_saved_model("my_model2.plk",housing_prepared)
    final_model = do_grid_search(housing_prepared, housing_labels)
    X_test = test.drop("median_house_value", axis=1)
    y_test = test["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_preds = final_model.predict(X_test_prepared)
    rmse = mean_squared_error(y_test,final_preds)
    print("Test Results (RMSE)")
    print(np.sqrt(rmse))


