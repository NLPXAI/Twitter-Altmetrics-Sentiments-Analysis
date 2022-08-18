# %% Imports
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time

# %% Imports
# Importing the dataset
# dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
# dataset.head()

X,y = shap.datasets.diabetes()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
X_train_summary = shap.kmeans(X_train, 10)

def print_accuracy(f):
    print("Root mean squared test error = {0}".format(np.sqrt(np.mean((f(X_test) - y_test)**2))))
    time.sleep(0.5) # to let the print get out before any progress bars

shap.initjs()


# %% Imports
from sklearn import linear_model
lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train, y_train)

print_accuracy(lin_regr.predict)
# %% Imports
ex = shap.KernelExplainer(lin_regr.predict, X_train_summary)
shap_values = ex.shap_values(X_test.iloc[0,:])
shap.force_plot(ex.expected_value, shap_values, X_test.iloc[0,:])

# %% Imports
shap_values = ex.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# %% Imports
shap.dependence_plot("bmi", shap_values, X_test)
# %% Imports
shap.force_plot(ex.expected_value, shap_values, X_test)
# %% Imports
from sklearn.ensemble import RandomForestRegressor
rforest = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
rforest.fit(X_train, y_train)
print_accuracy(rforest.predict)

# explain all the predictions in the test set
explainer = shap.TreeExplainer(rforest)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


