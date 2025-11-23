# src/build_rfm_model.py
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAW_PATH = os.path.join("data", "raw", "ecommerce_synthetic.csv")
PROC_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    df["order_date"] = pd.to_datetime(df["order_date"])
    snapshot_date = df["order_date"].max() + timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency=("order_date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("amount", "sum"),
        )
        .reset_index()
    )
    return rfm


def fit_kmeans(rfm: pd.DataFrame, n_clusters: int = 4):
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Basit KMeans (n_init default 10+)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, clusters)
    print(f"Silhouette score (k={n_clusters}): {sil:.3f}")

    rfm["cluster"] = clusters
    return rfm, scaler, kmeans


def add_cluster_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    # Clusterları "Monetary" ortalamasına göre sıralayıp anlamlı isim verelim
    cluster_stats = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .sort_values("Monetary", ascending=False)
    )
    ordered_clusters = list(cluster_stats.index)

    labels_map = {}
    for i, c in enumerate(ordered_clusters):
        if i == 0:
            labels_map[c] = "High-Value Loyal"
        elif i == 1:
            labels_map[c] = "Growth Potential"
        elif i == 2:
            labels_map[c] = "At-Risk"
        else:
            labels_map[c] = "Low-Value / Dormant"

    rfm["segment"] = rfm["cluster"].map(labels_map)
    return rfm


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}. Run generate_data.py first.")

    df = pd.read_csv(RAW_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows")

    rfm = build_rfm_table(df)
    print(f"RFM table shape: {rfm.shape}")

    rfm, scaler, kmeans = fit_kmeans(rfm, n_clusters=4)
    rfm = add_cluster_labels(rfm)

    # Kaydet
    rfm_path = os.path.join(PROC_DIR, "rfm_clusters.csv")
    rfm.to_csv(rfm_path, index=False)
    print(f"RFM + clusters saved to: {rfm_path}")

    joblib.dump(scaler, os.path.join(MODEL_DIR, "rfm_scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "rfm_kmeans.pkl"))
    print("Scaler and KMeans model saved under models/")


if __name__ == "__main__":
    main()
