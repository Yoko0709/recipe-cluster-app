import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Recipe Clustering App",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Recipe Clustering Web App")
st.markdown("""
Welcome to this interactive application! Use the sidebar to explore:

- 📊 Clustering Visualization with PCA  
- 🔥 Keyword Heatmaps for each Cluster  
- 🔍 Search Recipes by Keyword and Cluster Info  
""")

st.info("Use the left sidebar to start exploring the recipe data.")
