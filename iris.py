
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np

col_names=['sepal length','sepal width','petal length','petal width','iris'] 
df=pd.read_csv('C:\\Users\\deeksha.aneja\\Desktop\\jigsaw\\dsProjects\\iris\\iris_data.csv',names=col_names, header=None)
df.head()


# In[103]:


df.describe()


# In[108]:


df.groupby('iris').size()


# In[115]:


import seaborn as sns
import matplotlib.pyplot as plt

df.plot(kind='box', sharex=False, sharey=False)


# In[116]:


sns.boxplot( x=df["iris"], y=df["sepal length"], palette="Blues")


# In[120]:


sns.violinplot(data=df,x="iris", y="petal length")


# In[121]:


sns.pairplot(df, hue="iris")


# In[124]:


# updating the diagonal elements in a pairplot to show a kde(kernel density estimation)
sns.pairplot(df, hue="iris",diag_kind="kde")


# In[125]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(df.drop('iris',1))
le=LabelEncoder()
y=df['iris']
# print (y)
y=le.fit_transform(y)
print (y.shape)

# print (X.shape)
# y


# In[126]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
print(X_train.shape,y_train.shape)
# y_train


# In[127]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


lr=LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)
# lr=RandomForestClassifier()
print(X_train.shape,y_train.shape)
lr.fit(X_train,y_train)

y_predicted = lr.predict(X_test)
# Test Accuracy (RF)
print(metrics.accuracy_score(y_test, y_predicted))


# In[133]:


# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))

