
import os
from dotenv import load_dotenv
import snowflake.connector as s
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

con = s.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE')
)

q = 'select * from "CANCER"'
df = pd.read_sql(q, con)
con.close()
df.head()

df.head()

df.info()

df.isnull().sum()

df['DIAGNOSIS'] = df['DIAGNOSIS'].map({'M':1,'B':0})

df.head()

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=df.drop(['DIAGNOSIS'],axis=1)
y=df['DIAGNOSIS']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

lr=LogisticRegression()
lr.fit(X_train,y_train)

y_lrpred=lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_lrpred)
acc_lr

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

y_knnpred=knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_knnpred)
acc_knn

nb=GaussianNB()
nb.fit(X_train,y_train)

y_nbpred = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_nbpred)
acc_nb

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_dtpred = dt.predict(X_test)
acc_dt = accuracy_score(y_test,y_dtpred)
acc_dt

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_rfpred = rf.predict(X_test)
acc_rf = accuracy_score(y_test,y_rfpred)
acc_rf

svc = SVC()
svc.fit(X_train,y_train)

y_svcpred = svc.predict(X_test)
acc_svc = accuracy_score(y_test,y_svcpred)
acc_svc

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machine'],
    'Accuracy': [acc_lr, acc_knn, acc_nb, acc_dt, acc_rf, acc_svc]
})

models.sort_values(by='Accuracy', ascending=False)