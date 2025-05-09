import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.loader import load_processed_data, load_tfidf_vectorizer

st.set_page_config(layout="wide")
st.title("🔥 Keyword Heatmap by Cluster")

# Sidebar selection
keyword_type = st.radio("Select keyword type:", ["Ingredients", "Preparation Steps"])

# Load data
df = load_processed_data()
df["Cluster"] = df["Cluster"].astype(str)

if keyword_type == "Ingredients":
    vectorizer = load_tfidf_vectorizer("E:/tokyo/work/recipe-cluster-app/recipe-cluster-app/data/tfidf_ingredients.pkl")
    tfidf_matrix = vectorizer.transform(df["Ingredients_List"])
else:
    vectorizer = load_tfidf_vectorizer("E:/tokyo/work/recipe-cluster-app/recipe-cluster-app/data/tfidf_steps.pkl")
    tfidf_matrix = vectorizer.transform(df["Preparation_Steps"])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df["Cluster"] = df["Cluster"].values

# Calculate average weight of each word per cluster
cluster_keywords = tfidf_df.groupby("Cluster").mean().T
top_keywords = cluster_keywords.mean(axis=1).sort_values(ascending=False).head(20).index
cluster_keywords_top = cluster_keywords.loc[top_keywords]

# Draw heatmap
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    cluster_keywords_top,
    cmap="coolwarm",
    annot=False,
    linewidths=0.3,
    linecolor="gray",
    cbar_kws={"shrink": 0.6}
)
plt.title(f"Top 20 {keyword_type} Keywords by Cluster", fontsize=16)
plt.ylabel("Keywords")
plt.xlabel("Cluster")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
st.pyplot(fig, use_container_width=True)
