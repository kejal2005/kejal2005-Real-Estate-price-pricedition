# %% [markdown]
# ## Dragon Real Estate Price Predictor

# %%
import pandas as pd

# %%
housing = pd.read_csv("Real_estate_valuation_data_set.csv")

# %%
housing.head()

# %%
housing.info

# %%
housing.describe()

# %%
import matplotlib.pyplot as plt


# %%
import matplotlib.pyplot as plt 

# %%
housing.hist(bins = 50 , figsize = (20 , 15 ))

# %% [markdown]
# ## train-test spliting

# %%
# for learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size :]
    return data.iloc[train_indices], data.iloc[test_indices]

# %%
#train_set, test_set = split_train_test(housing, 0.2)

# %%
#print(f"Rows in train set: {len(train_set)}\nRows in test set : {len(test_set)}\n")

# %%
from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing, test_size=0.2 , random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set : {len(test_set)}\n")

# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['X4 number of convenience stores']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# %%
strat_test_set['X4 number of convenience stores'].value_counts()

# %%
strat_train_set['X4 number of convenience stores'].value_counts()

# %%
strat = strat_train_set.copy()

# %% [markdown]
# ## looking for correlations

# %%
corr_matrix = housing.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix

attributes = ["Y house price of unit area", "X4 number of convenience stores", 
              "X5 latitude", "X3 distance to the nearest MRT station"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# %%
housing.plot(kind="scatter" , x="X3 distance to the nearest MRT station" , y="Y house price of unit area" , alpha = 0.8)

# %% [markdown]
# ## attribute combinations 

# %%
housing["HOUSEPRICELATITUDE"] = housing["Y house price of unit area"] / housing["X5 latitude"]


# %%
housing["HOUSEPRICELATITUDE"]

# %%
housing.head()

# %%
corr_matrix = housing.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)

# %%
housing.plot(kind="scatter", x="HOUSEPRICELATITUDE" , y="Y house price of unit area")

# %%
housing["LATITUDELONGITUDE"] = housing["X5 latitude"] / housing["X6 longitude"]

# %%
housing["LATITUDELONGITUDE"]

# %%
housing.plot(kind="scatter", x="LATITUDELONGITUDE" , y="Y house price of unit area")

# %%
housing = strat_train_set.drop("Y house price of unit area", axis=1)
housing_lables = strat_train_set["Y house price of unit area"].copy()

# %% [markdown]
# ## missing attributs
# three ways to solve 

# %%
a = housing.dropna(subset = ["X3 distance to the nearest MRT station"])
a.shape


# %%
housing.drop("X3 distance to the nearest MRT station",axis=1)

# %%
median = housing["X3 distance to the nearest MRT station"].median()

# %%
housing = housing.fillna(housing.median())


# %%
median

# %%
housing.shape

# %%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# %%
imputer.statistics_

# %%
X = imputer.transform(housing)

# %%
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)


# %%
housing_tr.describe()

# %% [markdown]
# ## scikit learn
# 3 types of object

# %% [markdown]
# ## creating pipeline

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# %%
housing_num_tr = my_pipeline.fit_transform(housing)

# %%
housing_num_tr

# %% [markdown]
# ## selecting a model for real estates
# 

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  # Fixed the missing 'import'

# model = LinearRegression()
# model = DecisionTreeRegressor()  # ← parentheses are important!
model = RandomForestRegressor()  # Added parentheses to instantiate the model
model.fit(housing_num_tr, housing_lables)  # Fixed the typo: 'lables' → 'labels'


# %%
some_data = housing.iloc[:5]

# %%
some_lables = housing_lables.iloc[:5]

# %%
prepared_data = my_pipeline.transform(some_data)

# %%
model.predict(prepared_data)

# %%
list(some_lables)

# %% [markdown]
# ## evaluating the model

# %%
from sklearn.metrics import mean_squared_error

housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_lables, housing_predictions)
rmse = np.sqrt(mse)




# %%
mse

# %% [markdown]
# ## using better evaluation technique = cross validation
# 

# %%
# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, housing_num_tr, housing_lables, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)  # Negate since scores are negative MSE

# %%
rmse_scores

# %%
def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())  # Corrected the typo here
    print("Standard deviation: ", scores.std())


# %%
print_scores(rmse_scores)

# %% [markdown]
# ## Quiz: Convert this notebook onto a python file and the pipeline using Visual Studio Code

# %% [markdown]
# ## saving the model
# 

# %%
from joblib import dump, load
dump(model, 'Real_Estates.joblib')  # Corrected 'dumb' → 'dump'


# %% [markdown]
# ## testing the model

# %%
X_test = strat_test_set.drop("Y house price of unit area", axis=1)
Y_test = strat_test_set["Y house price of unit area"].copy()

X_test_prepared = my_pipeline.transform(X_test)

final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))


# %%
final_rmse

# %%
prepared_data[0]

# %% [markdown]
# ## using the model

# %%
from joblib import dump, load
import numpy as np 
model = load('Real_Estates.joblib')
features = np.array([ [1.33634919, -2.69827898,  10.18998546,  2.88477281, -0.37510886,
       -0.48993566, -1.38876478]])
model.predict(features)

# %%



