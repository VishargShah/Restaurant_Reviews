# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 23:18:09 2017

@author: Visharg Shah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data

#delimiter is tab so '\t' and we dont won't inverted comma so quoting = 3 where 3 is for inverted comma
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the text

import re #Clean Text
import nltk 
nltk.download('stopwords') #all the proposition etc. are in stopwords in nltk
#All proposition, articles like the in as a which doesn't give hint that review is postive or negative so remove that
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    #Will keep only text, remove numbers, punctuation, special char 
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) 
    #Will convert all to lower case
    review = review.lower()
    review = review.split() #split the whole review in the different words in a list
    ps = PorterStemmer() #Stemming will convert words like loved, loving into one same word like love which means the same
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    #Set is much faster than working with list
    review = ' '.join(review) #Different words are join together with sepreater as space
    corpus.append(review)

#Creating Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #will keep only 1500 ost frequent words, will reduce sparsity or use dimension reductionality
#you can convert lowercase, remove stopwords, use token pattern to keep only words here also im one go
X = cv.fit_transform(corpus).toarray() #Will create Sparse matrix
Y = dataset.iloc[:,1].values

#Using Naive Bayes for classification
#Splitting the data 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

#Fitting Naive Bayes on training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#Predicting the result
Y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

