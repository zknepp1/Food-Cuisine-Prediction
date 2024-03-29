# cs5293sp23-project2
FOOD CUISINE PREDICTION
ZACHARY KNEPP

# DESCRIPTION
THIS PROGRAM TAKES THE YUMMLY.JSON DATASET AND TRAINS A MULTICLASS CLASSIFIER TO PREDICT THE CUISINE TYPE BASED ON USER INPUT OF INGREDIENTS, AND WILL PROVIDE THE USER WITH RECIPES THAT CONTAIN SIMILAR INGREDIENTS. THE USER WILL PROVIDE THE PROGRAM WITH THE NUMBER OF SIMILAR RECIPES DESIRED, AND A LIST OF INGREDIENTS. THE PROGRAM WILL TAKE THE LIST OF INGREDIENTS, AND CLASSIFY THE INGREDIENTS WITH A RANDOM FOREST ALGORITHM. THE PREDICTED CLASSIFICATION IS THEN USED TO SUBSET A DATAFRAME THAT CONTAINS LIKE CUISINES. COSINE SIMILARITY IS CALCULATED ACROSS THE DATAFRAME WITH THE USER PROVIDED INGREDIENTS, AND THE TOP N CUISINES WITH THE HIGHEST COSINE SIMILARITY SCORES ARE SELECTED. FINALLY, RESULTS ARE PRINTED OUT.

# HOW TO RUN
IF THIS IS YOUR FIRST TIME DOWNLOADING AND RUNNING THE PROGRAM, THE PROGRAM MAY NEED TO BUILD THE MODEL. THIS WILL TAKE ABOUT 20 MINUTES. AFTERWORDS, THE PROGRAM WILL SAVE THE MODEL, AND REUSE IT. THIS WILL SAVE TIME FOR FUTURE RUNS.

MOVE INTO THE PROJECT FOLDER AND RUN THE FOLLOWING CODE:

pipenv run python project2.py --N 5 --ingredient "corn meal" --ingredient eggs --ingredient onions --ingredient peppers --ingredient salt --ingredient sage --ingredient "chicken broth"

--N # 
--N FOLLOWED BY A NUMBER REPRESENTS HOW MANY RECIPES YOU NEED PULLED. 

--ingredient ""
--ingredient FOLLOWED BY TEXT, OR TEXT IN QUOTES(FOR MULTI-WORD INGREDIENTS), REPRESENTS ONE SINGLE INGREDIENT.

https://user-images.githubusercontent.com/41703755/234102254-4c7c527c-4f23-40cf-a77a-f4d8a4d738ac.mp4



# FUNCTIONS
pull_keys(list_of_dictionaries):
PULLS CUISINES AND INGREDIENTS FROM A LIST OF DICTIONARIES AND RETURNS THE DATA IN LISTS

pull_key(d):
PULLS CUISINES AND INGREDIENTS FROM A SINGLE DICTIONARY, AND RETURNS THE KEYS

make_model():
RETURNS TFIDFVECTORIZER TO TRANSFORM THE DATA, AND A RANDOM FOREST MODEL WITH 100 TREES.

make_df():
RETURNS A DATAFRAME FROM THE YUMMLY.JSON FILE THAT CONSISTS OF CUISINES, ID, AND INGREDIENTS. RETURNS DATAFRAME

dict_to_vect(d, tv):
RETURNS A TRANSFORMED DICTIONARY TO VECTOR USING A TFIDFVECTORIZER

homemade_cosine_similarity(a,b):
SKLEARN COSINE_SIMILARITY() FUNCTION WAS GIVING ME TROUBLE SO I MADE MY OWN COSINE SIMILARITY SCORE.

print_results(df, pred, list_of_ingredients, N, score):
TAKES ALL PREDICTED AND STRUCTURED DATA. COMPUTES SIMILARITY SCORES AND PRINTS OUT THE TOP N RECIPES.

# BUGS AND ASSUMPTIONS
GITHUB DOES NOT ALLOW FILES OVER 100 MB, AND THE RANDOM FOREST MODEL SAT AROUND 116MB. TO WORK AROUND THIS, THE PROGRAM WILL CREATE THE MODEL UPON FIRST EXECUTION ~20 MINUTES. THEN PROGRAM SAVES THE MODEL LOCALLY FOR REUSE TO SAVE TIME.



ASSUMPTIONS MADE:
THE USER WILL HAVE THE RAM AVAILABLE TO MAKE THE MODEL.
THE USER WILL HAVE NO INPUT ERRORS.
ALL DATA FROM YUMMLY WERE REAL RECIPES.
THE USER KNOWS HOW TO SPELL INGREDIENTS (TURMERIC? TUMERIC?) (GARAM MASALA? GARAM MARSALA?).

BUGS:
IF THE USER DOES NOT HAVE ENOUGH RAM TO BUILD THE MODEL, THE PROGRAM WILL CRASH.
THE PROGRAM MAY CRASH IF YOU TRIED TO RETRIEVE A RIDICULOUS AMOUNT OF RECIPES. 
