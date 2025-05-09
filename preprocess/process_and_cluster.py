import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("recipe_data.csv")

# Data preprocessing
data.fillna({
    'Cuisine_Type': 'Unknown',
    'Difficulty_Level': 'Unknown',
    'User_Preferences': 'None'
}, inplace=True)
data['Cooking_Time_Minutes'].fillna(data['Cooking_Time_Minutes'].median(), inplace=True)
data['Calories_Per_Serving'].fillna(data['Calories_Per_Serving'].median(), inplace=True)
data['Cost_Per_Serving'].fillna(data['Cost_Per_Serving'].median(), inplace=True)

# Standardize numerical features
numeric_features = ['Cooking_Time_Minutes', 'Calories_Per_Serving', 'Cost_Per_Serving', 'Popularity_Score']
scaler = StandardScaler()
scaled_numeric_features = scaler.fit_transform(data[numeric_features])

# Extract text features using TF-IDF
tfidf_ingredients = TfidfVectorizer(max_features=500)
ingredients_features = tfidf_ingredients.fit_transform(data['Ingredients_List']).toarray()

tfidf_steps = TfidfVectorizer(max_features=500)
steps_features = tfidf_steps.fit_transform(data['Preparation_Steps']).toarray()

# Encode categorical features
categorical_features = ['Cuisine_Type', 'Difficulty_Level', 'User_Preferences']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categories = encoder.fit_transform(data[categorical_features])

# Combine numerical and text features
features = np.hstack([scaled_numeric_features, ingredients_features, steps_features, encoded_categories])

# Use the elbow method and silhouette score to select the optimal number of clusters
inertia = []
silhouette_scores = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, cluster_labels))

# Visualize the elbow method and silhouette scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Elbow Method (Inertia)')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Select the optimal number of clusters and perform clustering
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# Analyze clustering results
cluster_summary = data.groupby('Cluster').agg({
    'Cuisine_Type': lambda x: x.mode()[0],
    'Difficulty_Level': lambda x: x.mode()[0],
    'Cooking_Time_Minutes': 'mean',
    'Calories_Per_Serving': 'mean',
    'Popularity_Score': 'mean'
}).reset_index()

print("Cluster Summary:")
print(cluster_summary)

# Calculate silhouette score distribution
from sklearn.metrics import silhouette_samples
silhouette_vals = silhouette_samples(features, data['Cluster'])
data['Silhouette'] = silhouette_vals

# Visualize silhouette scores for each cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Silhouette', data=data)
plt.title('Silhouette Scores per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Silhouette Score')
plt.show()

# Calculate Calinski-Harabasz Index
from sklearn.metrics import calinski_harabasz_score
ch_score = calinski_harabasz_score(features, data['Cluster'])
print(f'Calinski-Harabasz Index: {ch_score:.2f}')

# Extract keywords for each cluster
def analyze_cluster_keywords(tfidf_matrix, feature_names, cluster_labels):
    cluster_keywords = pd.DataFrame(tfidf_matrix, columns=feature_names)
    cluster_keywords['Cluster'] = cluster_labels
    keyword_summary = cluster_keywords.groupby('Cluster').mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(keyword_summary.T, cmap='coolwarm', annot=False)
    plt.title("Cluster Keyword Distribution")
    plt.ylabel("Keywords")
    plt.xlabel("Clusters")
    plt.show()

# Analyze ingredient keywords for each cluster
analyze_cluster_keywords(ingredients_features, tfidf_ingredients.get_feature_names_out(), data['Cluster'])

# Analyze preparation step keywords for each cluster
analyze_cluster_keywords(steps_features, tfidf_steps.get_feature_names_out(), data['Cluster'])
