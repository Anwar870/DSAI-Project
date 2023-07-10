#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import seaborn as sns
data=pd.read_csv(r"C:\Users\Lenovo\Downloads\AI-Data.csv")
data.head(60)


# In[39]:


#Gender
print('percentage',data.gender.value_counts(normalize=True))
data.gender.value_counts(normalize=True).plot(kind='bar')


# In[40]:


#Raise Hands
print('percentage',data.raisedhands.value_counts(normalize=True))
data.raisedhands.value_counts(normalize=True).plot(kind='pie',figsize=(40,40),fontsize=30)


# In[41]:


#Parent Answering Survey
print('percentage',data.ParentAnsweringSurvey.value_counts(normalize=True))
data.ParentAnsweringSurvey.value_counts(normalize=True).plot(kind='pie')


# In[42]:


print('percentage',data.StudentAbsenceDays.value_counts(normalize=True))
data.StudentAbsenceDays.value_counts(normalize=True).plot(kind='pie')


# In[43]:


#ParentschoolSatisfaction
print('percentage',data.ParentschoolSatisfaction.value_counts(normalize=True))
data.ParentschoolSatisfaction.value_counts(normalize=True).plot(kind='pie')


# In[44]:


import matplotlib.pyplot as plt
fig, axarr =plt.subplots(2,2,figsize=[10,10])
sns.countplot(x='Class',data=data,ax=axarr[0,0])
sns.countplot(x='gender',data=data,ax=axarr[0,1])
sns.countplot(x='StageID',data=data,ax=axarr[1,0])
sns.countplot(x='Semester',data=data,ax=axarr[1,1])


# In[45]:


fig,axarr=plt.subplots(2,1,figsize=(10,10))
sns.countplot(x='Topic',data=data,ax=axarr[0])
sns.countplot(x='NationalITy',data=data,ax=axarr[1])


# In[46]:


fig,axarr=plt.subplots(2,2,figsize=(10,10))
sns.countplot(x='gender',hue='Class',data=data,ax=axarr[0,0],order=['M','F'],hue_order=['L','M','H'])
sns.countplot(x='gender',hue='Relation',data=data,ax=axarr[0,1],order=['M','F'])
sns.countplot(x='gender',hue='StudentAbsenceDays',data=data,ax=axarr[1,0],order=['M','F'])
sns.countplot(x='gender',hue='ParentAnsweringSurvey',data=data,ax=axarr[1,1],order=['M','F'])


# In[47]:


import seaborn as sns
fig,axarr=plt.subplots(2,1,figsize=(10,10))
sns.countplot(x='Topic',hue='gender',data=data,ax=axarr[0])


# In[48]:


fig,axarr=plt.subplots(2,1,figsize=(10,10))
sns.countplot(x='gender',hue='NationalITy',data=data,ax=axarr[1])
sns.countplot(x='Topic',hue='NationalITy',data=data,ax=axarr[0])


# In[49]:


fig,axarr=plt.subplots(2,1,figsize=(10,10))
sns.countplot(x='NationalITy',hue='Relation',data=data,ax=axarr[0])


# In[50]:


fig,axarr=plt.subplots(2,2,figsize=(10,10))
sns.barplot(x='Class',y='VisITedResources',data=data,ax=axarr[0,0])
sns.barplot(x='Class',y='AnnouncementsView',data=data,ax=axarr[0,1])
sns.barplot(x='Class',y='raisedhands',data=data,ax=axarr[1,0])
sns.barplot(x='Class',y='Discussion',data=data,ax=axarr[1,1])


# In[51]:


fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))
sns.barplot(x='gender',y='raisedhands',data=data,ax=axis1)
sns.barplot(x='gender',y='Discussion',data=data,ax=axis2)


# In[52]:


fig,(axis1,axis2)=plt.subplots(1,2,figsize=(20,15))
sns.swarmplot(x='gender',y='AnnouncementsView',data=data,ax=axis1)
sns.swarmplot(x='gender',y='raisedhands',data=data,ax=axis2)


# In[53]:


fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))
sns.boxplot(x='Class',y='Discussion',data=data,ax=axis1)
sns.boxplot(x='Class',y='VisITedResources',data=data,ax=axis2)


# In[54]:


fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))
sns.pointplot(x='Semester',y='VisITedResources',hue='gender',data=data,ax=axis1)
sns.pointplot(x='Semester',y='AnnouncementsView',hue='gender',data=data,ax=axis2)


# In[55]:


fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))
sns.regplot(x='raisedhands',y='VisITedResources',data=data,ax=axis1)
sns.regplot(x='AnnouncementsView',y='Discussion',data=data,ax=axis2)


# In[56]:


plot=sns.countplot(x='Class',hue='Relation',data=data,order=['L','M','H'],palette='Set1')
plot.set(xlabel='Class',ylabel='Count',title='Gender comparison')
plt.show()


# In[57]:


sns.pairplot(data,hue='Class')


# In[58]:


from sklearn.preprocessing import LabelEncoder
Features=data.drop('gender',axis=1)
target=data['gender']
label=LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
print(Features)


# In[59]:


Features=data.drop('Semester',axis=1)
target=data['Semester']
label=LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
print(Features)


# In[60]:


Features=data.drop('NationalITy',axis=1)
target=data['NationalITy']
label=LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
print(Features)


# In[61]:


Features=data.drop('Relation',axis=1)
target=data['Relation']
label=LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
print(Features)


# In[62]:


Features=data.drop('ParentAnsweringSurvey',axis=1)
target=data['ParentAnsweringSurvey']
label=LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
print(Features)


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(Features,target,test_size=0.2,random_state=52)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


# In[65]:


Logit_Model=LogisticRegression()
Logit_Model.fit(X_train,y_train)


# In[68]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
Prediction = Logit_Model.predict(X_test)
Score = accuracy_score(y_test,Prediction)
Report = classification_report(y_test,Prediction)


# In[69]:


print(Prediction)


# In[70]:


print(Score)


# In[71]:


print(Report)


# In[ ]:




