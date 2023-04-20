#! pip install scikit-learn
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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


###########################################

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

###########################################

# Pull key (singular) from dictionary
def pull_key(d):
  categories = []
  ingredients = []

  for key, value in d.items():
    if key == 'cuisine':
      categories.append(value)
    elif key == 'ingredients':
      ingredients.append(value)

  return categories, ingredients



###########################################



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

   km = KMeans(n_clusters = 20, random_state = 0, n_init = 'auto').fit(X_train)
   print(km.labels_)

   return km, tv

###########################################


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


###########################################
###########################################


def main():
   with open('yummly.json', 'r') as file:
      data = json.load(file)

   print(len(data))



   try:
     km = pickle.load(open("km_model.pickle", 'rb'))
     tv = pickle.load(open("vec.pickle", 'rb'))
     print('Model loaded successfully!')

   except:
     print('Could not load model...   :(')
     km, tv = make_model()
     print("Model has been created")

     km_model = 'km_model.pickle'
     vec = 'vec.pickle'
     pickle.dump(km, open(km_model, 'wb'))
     pickle.dump(tv, open(vec, 'wb'))
     print('Model has been saved! You can now skip the training, and use the model for future use')


   X = ["soy sauce noodles sesame seeds chicken"]
   tv_x = tv.transform(X)
   x1 = tv_x.toarray()
   y1 = 'italian'
   y_pred = km.predict(x1)

   print("y_pred1: ", y_pred)
   print("actual1: ", y1)




if __name__ == "__main__":
    main()
















