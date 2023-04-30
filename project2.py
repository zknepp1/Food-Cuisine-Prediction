#! pip install scikit-learn
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
from numpy import dot
from numpy.linalg import norm

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

# Returns the model and vectorizer
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
   rf = RandomForestClassifier(n_estimators=100)
   rf.fit(X, y)

   return rf, tv

###########################################

# Returns a datafram with cuisine, id, and ingredients
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

   df = pd.DataFrame()
   df['cuisine'] = categories
   df['id'] = id
   df['ingredients'] = ingredients
   return df

###########################################

# Takes ingredients and vectorizes them
def dict_to_vect(d, tv):
  categories, ingredients = pull_key(d)
  X = [" ".join(x) for x in ingredients]

  tv_x = tv.transform(X)
  tv_x = tv_x.toarray()
  y = categories

  return tv_x, y

###########################################

# I could not get the cosine similarity to work
# So i made my own
def homemade_cosine_similarity(a,b):
   cos_sim = dot(a, b)/(norm(a)*norm(b))
   return cos_sim


###########################################

# This function prints out all of the results
def print_results(df, pred, list_of_ingredients, N, score):
   pred_df = df[df['cuisine'] == pred[0]]
   ing = pred_df['ingredients'].tolist()

   X = [" ".join(x) for x in ing]

   cv = CountVectorizer()
   cv_matrix = cv.fit_transform(X)
   cv_array = cv_matrix.toarray()

   target = cv.transform(list_of_ingredients)
   target_array = target.toarray()
   target_array = np.squeeze(np.asarray(target_array))

   scores = []


   for i in cv_matrix:
      comp = i.toarray()
      comp = np.squeeze(np.asarray(comp))
      scores.append(homemade_cosine_similarity(target_array, comp))


   pred_df['score'] = scores
   sorted_df = pred_df.sort_values('score', ascending = False)
   top_n = sorted_df.iloc[:N]

   print('{')
   print('   cuisine:', pred)
   print('   score:', score)
   print('   closest: [')

   for index, row in top_n.iterrows():
      print('     {')
      print('       id:', row['id'])
      print('       score:', row['score'])
      print('     },')
   print('   ]')
   print('}')


###########################################
###########################################

def main():
   #Parsing the arguments given in the command line
   parser = argparse.ArgumentParser()
   parser.add_argument("--N")
   parser.add_argument("--ingredient", action='append')
   args = parser.parse_args()

   N = int(args.N)
   loi = " ".join(args.ingredient)
   list_of_ingredients = []
   list_of_ingredients.append(loi)

   #Making dataframe to store constructed data
   df = make_df()

   #The program will try to load existing models in the local folder
   try:
     rf = pickle.load(open("rf_model.pickle", 'rb'))
     tv = pickle.load(open("vec.pickle", 'rb'))
     print('Model loaded successfully!')

   # If the model is not in the local folder, then a new model will be made
   # This will take about 20 minutes
   except:
     print('The model could not be loaded.')
     print('A new model will be made, and saved for future use.')
     print('Please allow 20 minutes for the model to be made.')

     # Make model and vectorizer
     rf, tv = make_model()
     rf_model = 'rf_model.pickle'
     vec = 'vec.pickle'
     # Save model for future use
     pickle.dump(rf, open(rf_model, 'wb'))
     pickle.dump(tv, open(vec, 'wb'))

   # Transform the list of ingredients given by the user
   tv_x = tv.transform(list_of_ingredients)
   x1 = tv_x.toarray()

   # Predict and print results
   pred = rf.predict(x1)
   scores = rf.predict_proba(x1)
   score = np.max(scores)
   print_results(df, pred, list_of_ingredients, N, score)

if __name__ == "__main__":
    main()




