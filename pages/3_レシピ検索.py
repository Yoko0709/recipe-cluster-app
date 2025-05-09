import streamlit as st
import pandas as pd
from utils.loader import load_processed_data

st.set_page_config(layout="wide")
st.title("🔍 Recipe Search by Keyword")

# Load and preprocess data
df = load_processed_data()
df["Cluster"] = df["Cluster"].astype(str)

# Pre-compute cluster average statistics
cluster_avg = df.groupby("Cluster").agg({
    "Cooking_Time_Minutes": "mean",
    "Calories_Per_Serving": "mean",
    "Cost_Per_Serving": "mean",
    "Popularity_Score": "mean"
}).round(2)

# Text input
query = st.text_input("Enter a keyword to search for:", "")

if query:
    # Filter recipes containing keyword in either ingredients or steps
    results = df[
        df["Ingredients_List"].str.contains(query, case=False, na=False) |
        df["Preparation_Steps"].str.contains(query, case=False, na=False)
    ].copy()

    # Drop duplicate entries based on main fields
    results = results.drop_duplicates(
        subset=["Cluster", "Cuisine_Type", "Ingredients_List", "Preparation_Steps"]
    ).copy()

    # Create shortened previews for display
    results["Ingredients_Short"] = results["Ingredients_List"].apply(
        lambda x: str(x)[:80] + "." if pd.notnull(x) else ""
    )
    results["Steps_Short"] = results["Preparation_Steps"].apply(
        lambda x: str(x)[:80] + "." if pd.notnull(x) else ""
    )

    st.markdown(f"**Found {len(results)} matching (non-duplicate) recipes**")

    # Display each result in an expandable section
    for _, row in results.iterrows():
        cluster_id = row["Cluster"]
        with st.expander(f"🍽️ Cluster {cluster_id} | {row['Cuisine_Type']}"):
            st.markdown(f"**🧂 Ingredients Preview:** {row['Ingredients_Short']}")
            st.markdown(f"**📝 Steps Preview:** {row['Steps_Short']}")

            st.markdown(f"⏱️ Avg. Cooking Time: `{cluster_avg.loc[cluster_id, 'Cooking_Time_Minutes']}` min")
            st.markdown(f"🔥 Avg. Calories: `{cluster_avg.loc[cluster_id, 'Calories_Per_Serving']}` kcal")
            st.markdown(f"💰 Avg. Cost per serving: `{cluster_avg.loc[cluster_id, 'Cost_Per_Serving']}`")
            st.markdown(f"⭐ Avg. Popularity Score: `{cluster_avg.loc[cluster_id, 'Popularity_Score']}`")
else:
    st.info("Please enter a keyword to search for recipes.")

