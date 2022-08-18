# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Imports
# Importing the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.head()
# %% Imports
dataset.shape
# %% Imports
dataset.columns

# %% Imports
dataset.info()

# %% Imports
dataset = dataset.drop(['id'],axis='columns')
dataset
# %% Imports
dataset.isnull().sum()
# %% Imports
dataset['bmi'].dtype
# %% Imports
dataset['bmi'].fillna(float(dataset['bmi'].mean()), inplace=True)
dataset.head()
# %% Imports
dataset.isnull().sum()
# %% Imports
def genderToNumeric(gender):
    if gender == "Female":
        return 0
    else:
        return 1

def marriedToNumeric(marry):
    if marry == "Yes":
        return 1
    else:
        return 0   
    
def locationToNumeric(loc):
    if loc == "Urban":
        return 1
    else:
        return 0  

def worktypeToNumeric(worktype):
    if worktype == "Private":
        return 0
    elif worktype == "Govt_job":
        return 1  
    else: 
        return 2    
    
    
def smokeStatusToNumeric(smoking_status):
    if smoking_status == "Unknown":
        return 0
    elif smoking_status == "formerly smoked":
        return 1  
    elif smoking_status == "never smoked" :
        return 2 
    else: 
        return 3    
# %% Imports
dataset['gender']= dataset['gender'].apply(genderToNumeric) 
dataset['ever_married']= dataset['ever_married'].apply(marriedToNumeric) 
dataset['work_type']= dataset['work_type'].apply(locationToNumeric) 
dataset['Residence_type']= dataset['Residence_type'].apply(worktypeToNumeric) 
dataset['smoking_status']= dataset['smoking_status'].apply(smokeStatusToNumeric) 

# %% 
dataset.head()

# %% 
dataset['stroke'].value_counts()

# %% 
#sns.countplot(x=dataset['stroke'])
#plt.title('No of Patiesnts affected by stroke',fontsize=15)
#plt.show()
# %% Logistic Regression 

# %% 
X = dataset.drop(['stroke'],axis='columns')
#X.head(10)
len(X)

# %% 
y = dataset.stroke
#y.head(3)
len(y)

# %% 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# %% 
from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
lrmodel.fit(X_train, y_train)

# %% 
pred =  lrmodel.predict(X_test)
pred

# %% 
len(pred)

# %% 
lra=lrmodel.score(X_test, y_test)
lra
# %%
import lime_tabular

# %%
interpretor = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode='classification'
)

# %%
interpretor

# %%
X_test.iloc[2] 

# %%
y_test.iloc[2]

# %%
y_test

# %%
exp = interpretor.explain_instance(
    data_row= X_test.iloc[85], ##new data
    predict_fn=lrmodel.predict_proba
)

#exp.show_in_notebook(show_table=True)
exp

# %%
#exp.as_list()






# %%
