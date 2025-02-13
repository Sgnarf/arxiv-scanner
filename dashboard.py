import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

DB_FILE = "main_database.db"

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM arxiv_papers", conn)
    conn.close()
    df["published"] = pd.to_datetime(df["published"])
    df["date"] = df["published"].dt.date
    df["embedding"] = df["embedding"].apply(lambda x: np.frombuffer(x, dtype=np.float32))
    # Exclude clusters -1 and 0
    df = df[~df["cluster"].isin([-1,0])]
    return df


df = load_data()

st.title("ArXiv Paper Clustering Dashboard")

# Time Series (Cumulative)
st.subheader("Cumulative Papers per Cluster Over Time")
time_series_cumulative = df.groupby(["date", "cluster"]).size().unstack().fillna(0)
time_series_cumulative = time_series_cumulative.cumsum(axis=0)
st.line_chart(time_series_cumulative)


# Cluster Pie Chart
st.subheader("Cluster Distribution")
cluster_counts = df["cluster"].value_counts()
st.plotly_chart(px.pie(names=cluster_counts.index, values=cluster_counts.values, title="Cluster Distribution"))

# PCA Scatter Plot
st.subheader("Cluster Visualization (PCA)")
embeddings = np.vstack(df["embedding"].values)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalize(embeddings, norm="l2"))
df_pca = pd.DataFrame(pca_result, columns=["x", "y"])
df_pca["cluster"] = df["cluster"]
st.plotly_chart(px.scatter(df_pca, x="x", y="y", color=df_pca["cluster"].astype(str)))

# Word Cloud
st.subheader("Word Cloud for Cluster")
cluster_num = st.selectbox("Select Cluster:", df["cluster"].unique())
text = " ".join(df[df["cluster"] == cluster_num]["summary"].values)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
st.image(wordcloud.to_array(), use_container_width=True)
