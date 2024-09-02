''' 
Network analytics:-
Network analytics is the application of big data principles and tools to the management and security of data networks.


CRISP-ML(Q) process model describes six phases:
- Business and Data Understanding
- Data Preparation
- Model Building
- Model Evaluation and Hyperparameter Tuning
- Model Deployment
- Monitoring and Maintenance

Business Problem: There is a dataset consisting of information for the connecting routes. Create network analytics models on the dataset and measure degree centrality, degree of closeness centrality, and degree of in-between centrality.
Create a network using edge list matrix(directed only).

Objective(s): Maximize profitable route
Constraint(s): Minimize the transport cost

Success Criteria:
    a. Business: Increase the Number of tickets booking by 10% to 15%
    b. ML: 
    c. Economic: Additional revenue of $100K to $120K

Features:
 connecting routes=c("flights", " ID", "main Airport”, “main Airport ID", "Destination ","Destination  ID","haults","machinary")

Problem statement 2: 
Business Problem: There are three datasets given (Facebook, Instagram, and LinkedIn). Construct and visualize the following networks:
●	circular network for Facebook
●	star network for Instagram
●	star network for LinkedIn

Create a network using an adjacency matrix (undirected only). 
'''
 
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text

# Creating engine which link to SQL via python
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="1234", # passwrd
                               db="air_routes_db")) #database

# Reading data from loacal drive
connecting_routes = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Network Analytics/Network Analytics/connecting_routes.csv")

connecting_routes.head()

# Loading data into sql database
connecting_routes.to_sql('connecting_routes', con = engine, if_exists = 'replace', chunksize = 1000, index= False)



# Reading data from sql database
sql = 'select * from connecting_routes;'
connecting_routes = pd.read_sql_query(text(sql), con = engine.connect())

connecting_routes.head()

connecting_routes = connecting_routes.iloc[0:51, 1:8]
connecting_routes.columns


for_g = nx.Graph()
for_g = nx.from_pandas_edgelist(connecting_routes, source = 'source airport', 
                                target = 'destination apirport')


print(for_g)

# #  centrality:-
# 
# 
# **Degree centrality** is defined as the number of links incident upon a node (i.e., the number of ties that a node has). ... Indegree is a count of the number of ties directed to the node (head endpoints) and outdegree is the number of ties that the node directs to others (tail endpoints).
# 
# **Eigenvector Centrality** The adjacency matrix allows the connectivity of a node to be expressed in matrix form. So, for non-directed networks, the matrix is symmetric.Eigenvector centrality uses this matrix to compute its largest, most unique eigenvalues.
# 
# **Closeness Centrality** An interpretation of this metric, Centralness.
# 
# **Betweenness centrality** This metric revolves around the idea of counting the number of times a node acts as a bridge.


data = pd.DataFrame({"Degree": pd.Series(nx.degree_centrality(for_g)), 
                     "closeness":pd.Series(nx.closeness_centrality(for_g)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(for_g)),
                     "betweenness": pd.Series(nx.betweenness_centrality(for_g))}) 

data


# Visual Representation of the Network
connecting_routes1 = connecting_routes.iloc[0:51, 0:8]

for_g = nx.Graph()
for_g = nx.from_pandas_edgelist(connecting_routes1, source = 'source airport', 
                                target = 'destination apirport')

f = plt.figure()
pos = nx.spring_layout(for_g, k = 0.015)
nx.draw_networkx(for_g, pos, ax=f.add_subplot(111), node_size = 15, node_color = 'red')
plt.show()
#f.savefig("graph.png")


import networkx as nx
import matplotlib.pyplot as plt

# Define adjacency matrices for Facebook, Instagram, and LinkedIn
facebook_adjacency = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Network Analytics/Assignments/3e.Network Analytics/facebook.csv")

instagram_adjacency = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Network Analytics/Assignments/3e.Network Analytics/instagram.csv")

linkedin_adjacency = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Network Analytics/Assignments/3e.Network Analytics/linkedin.csv")

# Convert DataFrames to NumPy arrays
facebook_matrix = facebook_adjacency.values
instagram_matrix = instagram_adjacency.values
linkedin_matrix = linkedin_adjacency.values

# Create networks
facebook_network = nx.DiGraph(facebook_matrix)
instagram_network = nx.DiGraph(instagram_matrix)
linkedin_network = nx.DiGraph(linkedin_matrix)

data1 = pd.DataFrame({"Degree": pd.Series(nx.degree_centrality(facebook_network)), 
                     "closeness":pd.Series(nx.closeness_centrality(facebook_network)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(facebook_network)),
                     "betweenness": pd.Series(nx.betweenness_centrality(facebook_network))}) 

data1

data2 = pd.DataFrame({"Degree": pd.Series(nx.degree_centrality(instagram_network)), 
                     "closeness":pd.Series(nx.closeness_centrality(instagram_network)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(instagram_network)),
                     "betweenness": pd.Series(nx.betweenness_centrality(instagram_network))}) 

data2

data3 = pd.DataFrame({"Degree": pd.Series(nx.degree_centrality(linkedin_network)), 
                     "closeness":pd.Series(nx.closeness_centrality(linkedin_network)),
                     "eigenvector": pd.Series(nx.eigenvector_centrality(linkedin_network)),
                     "betweenness": pd.Series(nx.betweenness_centrality(linkedin_network))}) 

data3

# Plot networks
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
nx.draw_circular(facebook_network, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Facebook Circular Network')

plt.subplot(1, 3, 2)
pos = nx.spring_layout(instagram_network)
nx.draw(instagram_network, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Instagram Star Network')

plt.subplot(1, 3, 3)
pos = nx.spring_layout(linkedin_network)
nx.draw(linkedin_network, pos, with_labels=True, node_size=400, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('LinkedIn Star Network')
plt.axis('equal')  # Ensure the aspect ratio is equal to avoid stretching

plt.tight_layout()
plt.show()


