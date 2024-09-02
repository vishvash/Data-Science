from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "1234", db = "recommend_db"))

sql = 'select * from games' 
data = pd.read_sql_query(text(sql), engine.connect())

# Create user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='game', values='rating', fill_value=0)


# saving names to give dropdown in deployment
users_list = list(set(data.index))

# Load the saved model for processing
cosine_sim = joblib.load("matrix")

### Custom Function ###

# Function to get similar users
def get_similar_users(user_id, k):
    sim_users = cosine_sim[user_id - 1]  # User IDs start from 1
    similar_users = sorted(list(enumerate(sim_users, 1)), key=lambda x: x[1], reverse=True)
    similar_users = [user[0] for user in similar_users if user[0] != user_id]
    return similar_users[:k]

# Function to recommend DVDs to a user
def recommend_dvds(user_id, k):
    # if user_id not in user_item_matrix.index:
    #     print("User ID not found in the dataset.")
    #     return []

    similar_users = get_similar_users(user_id, k)
    user_ratings = user_item_matrix.iloc[user_id - 1]
    recommendations = []

    for sim_user in similar_users:
        sim_user_ratings = user_item_matrix.iloc[sim_user - 1]
        unrated_dvds = sim_user_ratings[sim_user_ratings == 0].index
        sim_user_sim = cosine_sim[user_id - 1][sim_user - 1]  # Similarity between users
        weighted_ratings = sim_user_ratings * sim_user_sim
        recommendations.extend(weighted_ratings[unrated_dvds].sort_values(ascending=False).index)

    return recommendations[:k]

######End of the Custom Function######    

app = Flask(__name__)

@app.route('/')
def home():
    #colours = ['Red', 'Blue', 'Black', 'Orange']
    return render_template("index.html", users_list = users_list)

@app.route('/guest', methods = ["post"])
def Guest():
    if request.method == 'POST' :
        mn = request.form["mn"]
        tp = request.form["tp"]
        top_n = recommend_dvds(int(mn), int(tp))
        top_n_df = pd.DataFrame(top_n , columns=['Games'])


        # Transfering the file into a database by using the method "to_sql"
        top_n_df.to_sql('top_games', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = top_n_df.to_html(classes = 'table table-striped')

        return render_template( "data.html", Y = "Results have been saved in your database", Z =  f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #5e617d;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}") 

if __name__ == '__main__':

    app.run(debug = False)