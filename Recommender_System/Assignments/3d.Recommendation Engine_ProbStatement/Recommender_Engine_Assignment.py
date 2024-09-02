# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:48:08 2024

@author: Lenovo
"""

'''
CRISP-ML(Q) process model describes six phases:
# - Business and Data Understanding
# - Data Preparation
# - Model Building
# - Model Evaluation and Hyperparameter Tuning
# - Model Deployment
# - Monitoring and Maintenance

# Business Problem: Build a recommender system with the given data using UBCF.
This dataset is related to the video gaming industry and a survey was conducted to build a 
recommendation engine so that the store can improve the sales of its gaming DVDs. A snapshot of the dataset is given below. Build a Recommendation Engine and suggest top-selling DVDs to the store customers.


# Objective(s): Maximize game effecient recommendation
# Constraint(s): Minimize the user's game selection time

Success Criteria:
    a. Business: Increase the Number of games purchase by 10% to 15%
    b. ML: 
    c. Economic: Additional revenue of $100K to $120K
    
Data Collection: 

Dimension: 5000 rows and 3 columns

Name of Feature	| Description  | Type         | Relevance
userId		    | User ID      | Nominal      | Relevant
game	        | Game title   | Nominal	  | Relevant
rating	        | User ratings | Quantitative | Relevant

'''

# Importing all required libraries, modules

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

games = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Recommender_System/Assignments/Dataset/Datasets_Recommendation Engine/game.csv", encoding = 'utf8')

# Database Connection
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "1234", db = "recommend_db"))

# Upload the Table into Database
games.to_sql('games', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# Read the Table (data) from MySQL database
sql = 'select * from games'
data = pd.read_sql_query(text(sql), con = engine.connect())

# Create DataFrame
df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot_table(index='userId', columns='game', values='rating', fill_value=0)

# Calculate cosine similarity between users
cosine_similar = cosine_similarity(user_item_matrix, user_item_matrix)

# Save the Pipeline for tfidf matrix
joblib.dump(cosine_similar, 'matrix')

# Load the saved model for processing
cosine_sim = joblib.load("matrix")

# Function to get similar users
def get_similar_users(user_id, k):
    sim_users = cosine_sim[user_id - 1]  # User IDs start from 1
    similar_users = sorted(list(enumerate(sim_users, 1)), key=lambda x: x[1], reverse=True)
    similar_users = [user[0] for user in similar_users if (user[0] != user_id) & (user[1] != 0 )]
    return similar_users

# Function to recommend DVDs to a user
def recommend_dvds(user_id, k):
    if user_id not in user_item_matrix.index:
        print("User ID not found in the dataset.")
        return []

    similar_users = get_similar_users(user_id, k)
    # user_ratings = user_item_matrix.iloc[user_id - 1]
    recommendations = []

    for sim_user in similar_users:
        sim_user_ratings = user_item_matrix.iloc[sim_user - 1]
        unrated_dvds = sim_user_ratings[sim_user_ratings == 0].index
        sim_user_sim = cosine_sim[user_id - 1][sim_user - 1]  # Similarity between users
        weighted_ratings = sim_user_ratings * sim_user_sim
        # recommendations.extend(weighted_ratings[exclude = unrated_dvds].sort_values(ascending=False).index)
        recommendations.extend(weighted_ratings.drop(index=unrated_dvds).sort_values(ascending=False).index)
    return list(set(recommendations[:k]))

# Example usage
user_id = 14  # Change user ID as per requirement
k = 13
recommended_dvds = recommend_dvds(user_id, k)
recommended_dvds
