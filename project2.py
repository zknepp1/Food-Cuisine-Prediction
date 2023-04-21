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
import argparse
import pandas as pd

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
   sub = data[0:35000]
   categories, ingredients = pull_keys(sub)
   X = [" ".join(x) for x in ingredients]
   tv = TfidfVectorizer(ngram_range=(1,2))
   tv_x = tv.fit_transform(X)
   tv_x = tv_x.toarray()

   X = tv_x
   y = categories
   km = KMeans(n_clusters = 20, random_state = 0, n_init = 'auto').fit(X)
   print(len(km.labels_))

   return km, tv

###########################################

def make_df():
   with open('yummly.json', 'r') as file:
      data = json.load(file)

   categories = []
   ingredients = []
   id = []
   for d in data:
     for key, value in d.items():
       if key == 'cuisine':
         categories.append(value)
       elif key == 'ingredients':
         ingredients.append(value)
       elif key == 'id':
         id.append(value)

   df = pd.DataFrame(data=categories)
   return df

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

   parser = argparse.ArgumentParser()
   parser.add_argument("--N")
   parser.add_argument("--ingredient", action='append')
   args = parser.parse_args()

   loi = "".join(args.ingredient)
   list_of_ingredients = []
   list_of_ingredients.append(loi)


   df = make_df()

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


   tv_x = tv.transform(list_of_ingredients)
   x1 = tv_x.toarray()
   y1 = 'italian'
   y_pred = km.predict(x1)

   print("y_pred1: ", y_pred)
   print("actual1: ", y1)




if __name__ == "__main__":
    main()




