import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle


def compute_rfm(df: pd.DataFrame, ref_date: datetime) -> pd.DataFrame:
    df = df.copy()
    df["transaction_date"] = df["transaction_timestamp"].dt.date

    rfm = (
        df.groupby("customer_id")
        .agg(
            last_purchase=("transaction_date", "max"),
            frequency=("transaction_id", "nunique"),
            monetary=("amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (ref_date.date() - rfm["last_purchase"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    return rfm


def build_rfm_clusters(transactions_path: str, processed_dir: str, n_clusters: int = 5):
    df = pd.read_csv(transactions_path, parse_dates=["transaction_timestamp"])
    ref_date = df["transaction_timestamp"].max() + pd.Timedelta(days=1)

    rfm = compute_rfm(df, ref_date)

    features = rfm[["recency", "frequency", "monetary"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    rfm["cluster"] = cluster_labels

    cluster_profile = (
        rfm.groupby("cluster")
        .agg(
            customers=("customer_id", "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    rfm["pca_1"] = X_pca[:, 0]
    rfm["pca_2"] = X_pca[:, 1]

    os.makedirs(processed_dir, exist_ok=True)
    rfm_path = os.path.join(processed_dir, "rfm_clusters.csv")
    profile_path = os.path.join(processed_dir, "cluster_profile.csv")
    artifacts_path = os.path.join(processed_dir, "rfm_artifacts.pkl")

    rfm.to_csv(rfm_path, index=False)
    cluster_profile.to_csv(profile_path, index=False)

    with open(artifacts_path, "wb") as f:
        pickle.dump({"scaler": scaler, "kmeans": kmeans, "pca": pca}, f)

    print(f"Saved RFM clusters to: {rfm_path}")
    print(f"Saved cluster profile to: {profile_path}")
    return rfm, cluster_profile


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_tx_path = os.path.join(base_dir, "data", "raw", "transactions.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    build_rfm_clusters(raw_tx_path, processed_dir, n_clusters=5)


if __name__ == "__main__":
    main()
