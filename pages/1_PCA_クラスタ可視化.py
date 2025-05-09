import streamlit as st
import plotly.express as px
from utils.loader import load_pca_data

st.set_page_config(layout="wide")
st.title("📊 PCA-based Clustering Visualization")

# Load PCA data
df = load_pca_data()
df["Cluster"] = df["Cluster"].astype(str)

# Cluster selection
cluster_options = sorted(df["Cluster"].unique().tolist())
selected = st.selectbox("Select a cluster to highlight (or view all)", ["All"] + cluster_options)

# Filter data
if selected != "All":
    filtered_df = df[df["Cluster"] == selected].copy()
else:
    filtered_df = df.copy()

# Plot
fig = px.scatter(
    filtered_df,
    x="PC1", y="PC2",
    color=filtered_df["Cluster"],
    hover_data=["Cuisine", "Main_Ingredients"],
    title="Recipe Clusters (PCA Projection)",
    width=1000,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
