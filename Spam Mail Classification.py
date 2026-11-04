#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib


# In[2]:


#converting csv file into tabular data
raw_data = pd.read_csv('mail_data.csv')


# In[3]:


print(raw_data)


# In[4]:


#replacing missing values with null string
data = raw_data.where((pd.notnull(raw_data)) , '')


# In[5]:


#displaying first 5 rows of the dataframe
data.head()


# In[6]:


#checking the number of rows and columns in dataframe
data.shape


# In[7]:


# Labelling spam mails as 0 and ham mails as 1
data['Category'] = data['Category'].map({
    'spam': 0,
    'ham': 1
})


# In[8]:


#separating the data as texts and labels
X = data['Message']
Y = data['Category']


# In[9]:


print(X)


# In[10]:


print(Y)


# In[14]:


#splitting the dta into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=3, test_size=0.2, stratify=Y
)

# In[15]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[16]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[19]:


#converting text data into feature vectors
feature_extraction = TfidfVectorizer(min_df=1 , stop_words='english' , lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#converting Y_train and Y_test values as integer
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[20]:


print(X_train_features)


# In[21]:


#training the model using logistic regression
model = LogisticRegression()

model.fit(X_train_features , Y_train)
# Saving the trained model and TF-IDF vectorizer
joblib.dump(model, 'spam_mail_model.pkl')
joblib.dump(feature_extraction, 'tfidf_vectorizer.pkl')


# In[22]:


#prediction on training data
training_data_prediction = model.predict(X_train_features)
training_data_accuracy = accuracy_score(Y_train , training_data_prediction)


# In[23]:


print('Accuracy on training data : ', training_data_accuracy)


# In[24]:


#prediction on testing data
testing_data_prediction = model.predict(X_test_features)
testing_data_accuracy = accuracy_score(Y_test , testing_data_prediction)


# In[25]:


print('Accuracy on testing data : ', testing_data_accuracy)


# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(Y_test, testing_data_prediction,
                            target_names=["Spam", "Ham"]))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, testing_data_prediction))


# In[33]:


#building a predictive system
input_mail = ["Customer service annoncement. You have a New Years delivery waiting for you. Please call 07046744435 now to arrange delivery"]

#converting text to feature vectors
input_mail_features = feature_extraction.transform(input_mail)

#predicting
prediction = model.predict(input_mail_features)

if prediction[0] == 0:
    print("Spam")
else:
    print("Ham")  







