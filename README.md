# cs5293sp23-project2
Food Cuisine Prediction

# DESCRIPTION
THIS PROGRAM TAKES THE YUMMLY.JSON DATASET AND TRAINS A MULTICLASS CLASSIFIER TO PREDICT THE CUISINE TYPE BASED ON USER INPUT OF INGREDIENTS, AND WILL PROVIDE THE USER WITH RECIPES THAT CONTAIN SIMILAR INGREDIENTS. THE USER WILL PROVIDE THE PROGRAM WITH THE NUMBER OF SIMILAR RECIPES DESIRED, AND A LIST OF INGREDIENTS. THE PROGRAM WILL TAKE THE LIST OF INGREDIENTS, AND CLASSIFY THE INGREDIENTS WITH A RANDOM FOREST ALGORITHM. THE PREDICTED CLASSIFICATION IS THEN USED TO SUBSET A DATAFRAME THAT CONTAINS LIKE CUISINES. COSINE SIMILARITY IS CALCULATED ACROSS THE DATAFRAME WITH THE USER PROVIDED INGREDIENTS, AND THE TOP N CUISINES WITH THE HIGHEST COSINE SIMILARITY ARE SELECTED. RESULTS ARE THEN PRINTED OUT.
