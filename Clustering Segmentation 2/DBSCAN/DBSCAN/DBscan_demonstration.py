# DBScan demonstration on Spherical Data. 
# Compare the applications of Agglomerative, Kmeans, and DBScan clustering techniques

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Clustering Segmentation 2/DBSCAN/DBSCAN/Dbscan_spherical_data.csv")
df

plt.scatter(df.x, df.y)
plt.show()

# Agglomerative Hierarchical Clustering
ac = AgglomerativeClustering(5, linkage = 'average')
ac_clusters = ac.fit_predict(df)
plt.figure(1)
plt.title("Clusters from Agglomerative Clustering")
plt.scatter(df.x, df.y, c = ac_clusters, s = 50, cmap = 'tab20b')
plt.show()


# KMeans
km = KMeans(5)
km_clusters = km.fit_predict(df)
plt.figure(2)
plt.title("Clusters from K-Means")
plt.scatter(df.x, df.y, c = km_clusters, s = 50, cmap = 'tab20b')
plt.show()


# DBSCAN 
db = DBSCAN(eps = 0.2, min_samples = 11).fit(df)
labels = db.labels_
labels
plt.figure(3)
plt.scatter(df.x, df.y, c = labels, cmap = 'tab20b')
plt.title("DBSCAN from Scratch Performance")
plt.show()
