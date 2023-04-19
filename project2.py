#! pip install scikit-learn
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import gc
import pickle


nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))




# Pull keys from dictionary
def pull_keys(list_of_dictionaries):
  categories = []
  ingredients = []
  for d in list_of_dictionaries:
    for key, value in d.items():
      if key == 'cuisine':
        categories.append(value)
      elif key == 'ingredients':
        ingredients.append(value)

  return categories, ingredients



# Pull keys from dictionary
def pull_key(d):
  categories = []
  ingredients = []

  for key, value in d.items():
    if key == 'cuisine':
      categories.append(value)
    elif key == 'ingredients':
      ingredients.append(value)

  return categories, ingredients







def make_model():
   with open('yummly.json', 'r') as file:
      data = json.load(file)

   sub = data[0:30000]
   categories, ingredients = pull_keys(sub)
   X = [" ".join(x) for x in ingredients]
   tv = TfidfVectorizer(ngram_range=(1,2))
   tv_x = tv.fit_transform(X)
   tv_x = tv_x.toarray()

   X = tv_x
   y = categories
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
   #train, test = train_test_split(df, test_size=0.2)
   
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators = 30, random_state=0)
   # training a linear SVM classifier
   rf.fit(X, y)
   # model accuracy for X_test  
   y_pred = rf.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print("Accuracy:", accuracy)

   return rf, tv




def dict_to_vect(d, tv):
  categories, ingredients = pull_key(d)
  for i in ingredients:
    print(i)
  print(categories)
  X = [" ".join(x) for x in ingredients]
  
  tv_x = tv.transform(X)
  tv_x = tv_x.toarray()
  y = categories 

  return tv_x, y




def main():
   with open('yummly.json', 'r') as file:
      data = json.load(file)

   test1 = data[30001]
   test2 = data[30102]
   test3 = data[31003]
   test4 = data[31114]


   try:
     rf = pickle.load(open("rf_model.pickle", 'rb'))
     tv = pickle.load(open("vec.pickle", 'rb'))
     print('Model loaded successfully!')

   except:
     print('Could not load model...   :(')
     rf, tv = make_model()
     print("Model has been created")

     rf_model = 'rf_model.pickle'
     vec = 'vec.pickle'
     pickle.dump(rf, open(rf_model, 'wb'))
     pickle.dump(tv, open(vec, 'wb'))
     print('Model has been saved!')

   
   #x1, y1 = dict_to_vect(test1, tv)
   #x2, y2 = dict_to_vect(test2, tv)
   #x3, y3 = dict_to_vect(test3, tv)
   #x4, y4 = dict_to_vect(test4, tv)

   X = ["ground beef tomato soup noodles corn salt pepper good shit minestrone soup bloody mary mix"]
   tv_x = tv.transform(X)
   x1 = tv_x.toarray()
   y1 = 'italian'
   y_pred1 = rf.predict(x1)
   #y_pred2 = rf.predict(x2)
   #y_pred3 = rf.predict(x3)
   #y_pred4 = rf.predict(x4)

   print("y_pred1: ", y_pred1)
   print("actual1: ", y1)

   #print("y_pred2: ", y_pred2)
   #print("actual2: ", y2)

   #print("y_pred3: ", y_pred3)
   #print("actual3: ", y3)

   #print("y_pred4: ", y_pred4)
   #print("actual4: ", y4)


   #x = loaded_model.transform(x)
   #y_pred = loaded_model.predict(x)
   
   #print(y_pred)




if __name__ == "__main__":
    main()
















