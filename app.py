import os
import pandas as pd
import streamlit as st
import plotly.express as px


BASE_DIR = os.path.dirname(__file__)


@st.cache_data
def load_data():
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    rfm_path = os.path.join(processed_dir, "rfm_clusters.csv")
    profile_path = os.path.join(processed_dir, "cluster_profile.csv")

    rfm = pd.read_csv(rfm_path)
    profile = pd.read_csv(profile_path)
    return rfm, profile


def main():
    st.set_page_config(page_title="Customer Segmentation – RFM Clustering", layout="wide")
    st.title("🧩 Customer Segmentation – RFM + K-Means")
    st.write(
        "RFM-based customer segmentation using synthetic transaction data. "
        "Each customer is assigned to a cluster based on Recency, Frequency, and Monetary value."
    )

    rfm, profile = load_data()

    st.subheader("Cluster Overview")
    st.dataframe(profile.style.format({"avg_recency": "{:.1f}", "avg_frequency": "{:.2f}", "avg_monetary": "{:.2f}"}))

    st.subheader("RFM Scatter (PCA)")
    fig = px.scatter(
        rfm,
        x="pca_1",
        y="pca_2",
        color="cluster",
        hover_data=["customer_id", "recency", "frequency", "monetary"],
        title="Customer Clusters in 2D (PCA of RFM)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Drill-Down")
    clusters = sorted(rfm["cluster"].unique().tolist())
    selected_cluster = st.selectbox("Select cluster", options=clusters, index=0)

    segment = rfm[rfm["cluster"] == selected_cluster].copy()
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers in cluster", len(segment["customer_id"].unique()))
    col2.metric("Avg recency (days)", f"{segment['recency'].mean():.1f}")
    col3.metric("Avg monetary", f"{segment['monetary'].mean():.2f}")

    st.write("Sample customers in this cluster:")
    st.dataframe(segment[["customer_id", "recency", "frequency", "monetary"]].head(20))


if __name__ == "__main__":
    main()
