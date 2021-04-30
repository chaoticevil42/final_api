# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:29:47 2021

@author: kzani
"""
import pandas as pd 
import numpy as np
missing_value=["Undefined"]
data = pd.read_csv("hotel_bookings.csv", na_values=missing_value) 
data.head()

#Converting the object datatype of "reservation_status_date" to datetime datatype

data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])
type(data['reservation_status_date'][0])

# Creating a new column by combining the year, month and date of arrival together.

data['arrival_date'] = pd.to_datetime(data.arrival_date_year.astype(str) + '/' + data.arrival_date_month.astype(str) + '/' + data.arrival_date_day_of_month.astype(str))

# data['arrival_date'] # check the new timestamp

#Checking how many missing values each column contains

np.sum(data.isnull())


# find the indicies of the missing data and use the 50% criteria meaning
# if there is 50% or more of data missing then we will remove the feature from 
# the data array it can be argued that a value closer to 30 % may be a better
# rule of thumb. 

my_criteria = 0.5

for col in data.columns:
    if np.sum(data[col].isnull())>(data.shape[0] * my_criteria):
        data.drop(columns=col, inplace=True, axis=1)

for col in data.columns:
    if np.sum(data[col].isnull())>(data.shape[0] * 0.5):
        data.drop(columns=col, inplace=True, axis=1)
print(data.shape)

# We combined the date above so we can drop the columns associated with the dates

data.drop(columns=["arrival_date_week_number", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"],
           inplace=True, axis=1)

print(data.shape)

# now lets remove the rows with missing values

data.dropna(subset=["agent"], inplace=True)
print(data.shape)

data["children"].fillna(value = data["children"].mean(), inplace=True)
data["children"] = data["children"].apply(np.floor)
print(f"Total missing values in children column after filling = {np.sum(data.children.isnull())}")



colstofill=["market_segment", "distribution_channel", "meal", "country"]
print("Number of missing values are")
for x in colstofill:
    data[x].fillna(method="bfill", inplace=True)
    print(f"{x}: {np.sum(data[x].isnull())}")

np.sum(data.isnull())

# Lets get X and Y we are trying to predict a cancellation please review the 
# features that we are passing now. 

Y = data['is_canceled']
Y = Y[1:]
X = data.drop(['is_canceled'], axis=1)
X = X[1:]

from sklearn import preprocessing
enc = preprocessing.OrdinalEncoder()
enc.fit(X)
X_new = enc.transform(X)
X_new.shape

# Need to transform cetegorical data to something the model can use

#from sklearn import preprocessing
#enc = preprocessing.OrdinalEncoder()
#col_to_trans = ['meal' , 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status'  ]
#enc.fit(X[col_to_trans])
#X_new = enc.transform(X[col_to_trans])
#X_new.shape


# Now we need to look at Y
# In reality a roughly 60/40 split is not awful and very representative of 'real' world data but we will balance it anyway.

from collections import Counter

original_counts = Counter(Y)
print(original_counts)

# ~62K to 40k


# now lets balance the data
from imblearn.over_sampling import SMOTE
over = SMOTE()
X_new, Y_new = over.fit_resample(X_new,Y)
new_counts = Counter(Y_new)
print(new_counts)

# The deprication warning should be resolved. https://github.com/skorch-dev/skorch/issues/612

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization with out the use of SMOTE.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_new, Y_new, test_size=0.2, random_state=0)

pca = PCA()

logistic = LogisticRegression(max_iter=100000, tol=0.1)

pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# This grid search is exhaustive and time consuming. 

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [2, 4, 8, 10,12,14, 16, 20, 25],
    'logistic__C': np.logspace(-4, 4, 6),
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(x_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# This section is about using PCA to find the optimal combination of features
# https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html

pca.fit(x_train)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(np.arange(1, pca.n_components_ + 1),
         pca.explained_variance_ratio_, '+', linewidth=2)
ax0.set_ylabel('PCA explained variance ratio')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.xlim(-1, 70)

plt.tight_layout()

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_new, Y_new, test_size=0.2, random_state=0)

pca = PCA(n_components=14)
stdscl = StandardScaler()
logistic = LogisticRegression(C = 0.15848931924611143, max_iter=100000, tol=0.1)


pipe = Pipeline(steps=[('standardscalar', stdscl),('pca', pca), ('logistic', logistic)])

print(pipe.score(x_test,y_test))

my_prediction = pipe.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, my_prediction, target_names=['0', '1']))

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, my_prediction)

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = pipe.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, my_prediction)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
