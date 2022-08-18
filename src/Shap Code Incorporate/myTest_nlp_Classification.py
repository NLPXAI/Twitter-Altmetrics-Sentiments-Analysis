# %% Imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import shap
import time
import transformers

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# %% Imports
# Importing the dataset

# dataset = pd.read_csv('C:/Users/Asif Raza/Desktop/4th semester/explainable ai/Code/Lime Code Incorporate/lime/healthcare-dataset-stroke-data.csv')
# dataset.head()

reviews_train = pd.read_csv('C:/Users/Asif Raza/Desktop/4th semester/explainable ai/Code/Lime Code Incorporate/lime/Dataset/reviews_train.csv')
reviews_test = pd.read_csv('C:/Users/Asif Raza/Desktop/4th semester/explainable ai/Code/Lime Code Incorporate/lime/Dataset/reviews_test.csv')
# %% Imports
print(reviews_train.shape)
print(reviews_test.shape)
# %% Imports
#print(dataset.columns)
reviews_train = reviews_train.head(10000)
reviews_train.count

# %% Imports
#print(dataset.info())
cv = CountVectorizer(binary=True, stop_words='english')
cv.fit(reviews_train.review)
X = cv.transform(reviews_train.review)
X_test = cv.transform(reviews_test.review)

# %% Imports
#dataset = dataset.drop(['id'],axis='columns')
#dataset
# build a random forest

X_train, X_val, y_train, y_val = train_test_split(X, reviews_train.label, train_size = 0.8)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)

# %% 
print ("Accuracy: %s"  % accuracy_score(y_val, rf.predict(X_val)))


# %% 
# from lime import lime_text
#from sklearn.pipeline import make_pipeline
#c = make_pipeline(cv, rf)
classifier = transformers.pipeline('sentiment-analysis', return_all_scores=True)
classifier(short_data[:2])

# %% 
# from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=[0,1])
# %% 
idx =64
exp = explainer.explain_instance(reviews_test.review[idx], c.predict_proba, num_features=10)
#print('True class: %s' % reviews_test.label[idx])
exp.show_in_notebook(text=True)
exp
# %% 

# %% 



# %%
