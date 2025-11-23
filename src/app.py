# app.py
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

RFM_PATH = os.path.join("data", "processed", "rfm_clusters.csv")
SCALER_PATH = os.path.join("models", "rfm_scaler.pkl")
KMEANS_PATH = os.path.join("models", "rfm_kmeans.pkl")


@st.cache_data
def load_rfm():
    if not os.path.exists(RFM_PATH):
        raise FileNotFoundError("rfm_clusters.csv not found. Run src/build_rfm_model.py first.")
    return pd.read_csv(RFM_PATH)


@st.cache_resource
def load_models():
    if not (os.path.exists(SCALER_PATH) and os.path.exists(KMEANS_PATH)):
        raise FileNotFoundError("Model files not found. Run src/build_rfm_model.py first.")
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return scaler, kmeans


def main():
    st.set_page_config(
        page_title="Customer Segmentation – RFM + Clustering",
        page_icon="🧊",
        layout="wide",
    )

    st.title("🧊 Customer Segmentation – RFM + KMeans")
    st.markdown(
        """
Interactive customer segmentation demo on a synthetic e-commerce dataset.  
We compute **RFM features** (Recency, Frequency, Monetary), apply **K-Means clustering**,  
and explore segments through an interactive dashboard.
"""
    )

    try:
        rfm = load_rfm()
        scaler, kmeans = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")

    segments = sorted(rfm["segment"].unique())
    selected_segments = st.sidebar.multiselect(
        "Customer segments", segments, default=segments
    )

    recency_max = int(rfm["Recency"].max())
    freq_max = int(rfm["Frequency"].max())
    mon_max = int(rfm["Monetary"].max())

    recency_range = st.sidebar.slider(
        "Recency (days)",
        min_value=0,
        max_value=recency_max,
        value=(0, recency_max),
    )
    freq_range = st.sidebar.slider(
        "Frequency (orders)",
        min_value=1,
        max_value=freq_max,
        value=(1, freq_max),
    )
    mon_range = st.sidebar.slider(
        "Monetary (total spend)",
        min_value=float(0),
        max_value=float(mon_max),
        value=(0.0, float(mon_max)),
    )

    filtered = rfm[
        (rfm["segment"].isin(selected_segments))
        & (rfm["Recency"].between(recency_range[0], recency_range[1]))
        & (rfm["Frequency"].between(freq_range[0], freq_range[1]))
        & (rfm["Monetary"].between(mon_range[0], mon_range[1]))
    ]

    st.subheader("📊 Segment Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(filtered):,}")
    col2.metric("Avg Recency (days)", f"{filtered['Recency'].mean():.1f}")
    col3.metric("Avg Frequency", f"{filtered['Frequency'].mean():.1f}")
    col4.metric("Avg Monetary", f"${filtered['Monetary'].mean():.2f}")

    # Cluster scatter
    st.markdown("### RFM Scatter (colored by segment)")
    fig_scatter = px.scatter(
        filtered,
        x="Recency",
        y="Monetary",
        color="segment",
        size="Frequency",
        hover_data=["customer_id", "cluster"],
        height=450,
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Segment bar chart
    st.markdown("### Customers per segment")
    seg_counts = filtered["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    fig_bar = px.bar(seg_counts, x="segment", y="count", text="count", height=350)
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🎯 Segment a New Customer")

    c1, c2 = st.columns(2)
    with c1:
        recency_input = st.slider("Recency (days since last order)", 0, recency_max, 60)
        freq_input = st.slider("Frequency (number of orders)", 1, freq_max, 5)
        mon_input = st.slider(
            "Monetary (total spend)", 0.0, float(mon_max), float(mon_max / 4), step=10.0
        )

    with c2:
        st.write("Describe a hypothetical customer and predict their segment.")

        if st.button("Predict segment"):
            X = np.array([[recency_input, freq_input, mon_input]])
            X_scaled = scaler.transform(X)
            cluster = int(kmeans.predict(X_scaled)[0])

            # cluster → segment label
            cluster_map = (
                rfm.groupby("cluster")["segment"]
                .agg(lambda x: x.value_counts().index[0])
                .to_dict()
            )
            segment = cluster_map.get(cluster, f"Cluster {cluster}")

            st.success(f"Predicted cluster: **{cluster}**  \nPredicted segment: **{segment}**")

            # Basit açıklama
            if "High-Value" in segment:
                msg = "This looks like a high-value, loyal customer. Focus on retention perks and upsell."
            elif "Growth" in segment:
                msg = "Good engagement — nurture with targeted campaigns to increase frequency and spend."
            elif "At-Risk" in segment:
                msg = "Warning signs of churn. Consider win-back campaigns or discounts."
            else:
                msg = "Low engagement. Use education/onboarding content and light-touch promotions."

            st.info(msg)


if __name__ == "__main__":
    main()
