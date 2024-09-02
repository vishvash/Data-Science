'''
CRISP-ML(Q) process model describes six phases:
# - Business and Data Understanding
# - Data Preparation
# - Model Building
# - Model Evaluation and Hyperparameter Tuning
# - Model Deployment
# - Monitoring and Maintenance

# Business Problem: One of the OTT platform is facing the issue with viewership.
    The total screen time of the customers is low, which is causing the revenue to decline.

# Objective(s): Maximize customers screentime
# Constraint(s): Minimize the marketing budget

Success Criteria:
    a. Business: Increase the viewership by 10% to 15%
    b. ML: 
    c. Economic: Additional revenue of $100K to $120K
    
    Data Collection: 
        Dimension: 12294 rows and 7 columns
        1. anime_id
        2. name
        3. genre
        4. type
        5. episodes
        6. rating
        7. members   
'''

# Importing all required libraries, modules
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
# term frequency - inverse document frequency is a numerical statistic 
# that is intended to reflect how important a word is to document in a 
# collecion or corpus

# from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# import Dataset 
anime = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Recommender_System/Recommender_System/anime.csv", encoding = 'utf8')

# Database Connection
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "1234", db = "recommend_db"))


# Upload the Table into Database
anime.to_sql('anime', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# Read the Table (data) from MySQL database
sql = 'select * from anime'
anime = pd.read_sql_query(text(sql), con = engine.connect())

# Check for Missing values
anime["genre"].isnull().sum()

# Impute the Missing values in 'genre' column for a movie with 'General' category
anime["genre"] = anime["genre"].fillna("General")

# Create a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")   # taking stop words from tfidf vectorizer 

# Transform a count matrix to a normalized tf-idf representation
tfidf_matrix = tfidf.fit(anime.genre)  

# Save the Pipeline for tfidf matrix
joblib.dump(tfidf_matrix, 'matrix')

os.getcwd()

# Load the saved model for processing
mat = joblib.load("matrix")

tfidf_matrix = mat.transform(anime.genre)

tfidf_matrix.shape 

# cosine(x, y)= (x.y) / (||x||.||y||)
# Computing the cosine similarity on Tfidf matrix

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Create a mapping of anime name to index number
anime_index = pd.Series(anime.index, index = anime['name']).drop_duplicates()

# Example
anime_id = anime_index["Assassins (1995)"]
anime_id

# anime_id = 1465

topN = 5

# Custom function to find the TopN movies to be recommended

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    anime_id = anime_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that anime
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Getting the movie index 
    anime_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    anime_similar_show = pd.DataFrame(columns = ["name", "Score"])
    anime_similar_show["name"] = anime.loc[anime_idx, "name"]
    anime_similar_show["Score"] = anime_scores
    # anime_similar_show.drop(anime_similar_show[anime_similar_show['index'] == anime_id].index, inplace = True)
    anime_similar_show.drop(index = anime_id, inplace = True)
    anime_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    return(anime_similar_show)
    # return(anime_similar_show.iloc[1:, ])

# Call the custom function to make recommendations
rec = get_recommendations("No Game No Life Movie", topN = 10)
rec

