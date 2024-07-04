"""
Script for training and evaluating multiple machine learning models
for movement classification based on coordinate data.

1. Data Preparation:
   - Reads movement coordinate data from 'movement_classes_coordinates.csv'.
   - Filters and prepares data for training and testing.

2. Model Training:
   - Uses pipelines to train Logistic Regression, Ridge Classifier,
     Random Forest, and Gradient Boosting models.
   - Evaluates each model's accuracy on a test set.

3. Model Persistence:
   - Saves the trained Random Forest model to 'movement_classification_model.pkl'
     using pickle for future use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('movement_classes_coordinates.csv')

df.head()

df.tail()

df[df['class'] == 'Arm position is down']

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=int(10e10))),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier(max_iter=int(10e10))),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

fit_models['rc'].predict(X_test)

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

print(fit_models['rf'].predict(X_test))
print(y_test)

with open('movement_classification_data.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)




