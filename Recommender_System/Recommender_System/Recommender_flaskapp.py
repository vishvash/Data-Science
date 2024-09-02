from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "root", pw = "1234", db = "recommend_db"))

sql = 'select * from anime' 
anime = pd.read_sql_query(text(sql), engine.connect())

# anime = pd.read_csv("anime.csv", encoding = 'utf8')
anime["genre"] = anime["genre"].fillna(" ")

# saving names to give dropdown in deployment
movies_list = anime['name'].to_list()

tfidf = joblib.load('matrix')

tfidf_matrix = tfidf.transform(anime.genre)

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

anime_index = pd.Series(anime.index, index = anime['name']).drop_duplicates()

### Custom Function ###
def get_recommendations(Name, topN):    
    anime_id = anime_index[Name]
    
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Getting the movie index 
    anime_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    anime_similar_show = pd.DataFrame(columns=["name", "Score"])
    anime_similar_show["name"] = anime.loc[anime_idx, "name"]
    anime_similar_show["Score"] = anime_scores
    anime_similar_show.drop(index = anime_id, inplace = True)
    anime_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis = 1, inplace = True)
    return(anime_similar_show)

######End of the Custom Function######    

app = Flask(__name__)

@app.route('/')
def home():
    #colours = ['Red', 'Blue', 'Black', 'Orange']
    return render_template("index.html", movies_list = movies_list)

@app.route('/guest', methods = ["post"])
def Guest():
    if request.method == 'POST' :
        mn = request.form["mn"]
        tp = request.form["tp"]
        
        top_n = get_recommendations(mn, topN = int(tp))

        # Transfering the file into a database by using the method "to_sql"
        top_n.to_sql('top_10', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = top_n.to_html(classes = 'table table-striped')

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