import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def simulate_transactions(n_customers: int = 4000, avg_tx_per_customer: int = 8) -> pd.DataFrame:
    customers = [f"C{idx:05d}" for idx in range(1, n_customers + 1)]
    countries = ["DE", "IT", "CH", "TR", "US", "FR", "NL", "SE"]
    channels = ["web", "mobile_app", "store", "partner"]

    today = datetime(2025, 1, 1)
    start_date = today - timedelta(days=365)

    records = []
    tx_id = 1

    for c in customers:
        n_tx = max(1, np.random.poisson(lam=avg_tx_per_customer))
        base_country = np.random.choice(countries)
        for _ in range(n_tx):
            days_offset = np.random.randint(0, 365)
            ts = start_date + timedelta(days=int(days_offset), hours=np.random.randint(0, 24))
            channel = np.random.choice(channels, p=[0.5, 0.3, 0.15, 0.05])
            base_amount = {
                "web": 40.0,
                "mobile_app": 30.0,
                "store": 60.0,
                "partner": 80.0,
            }[channel]
            amount = base_amount * np.random.uniform(0.5, 1.8)
            records.append(
                {
                    "transaction_id": f"T{tx_id:07d}",
                    "customer_id": c,
                    "country": base_country,
                    "channel": channel,
                    "transaction_timestamp": ts,
                    "amount": round(float(amount), 2),
                }
            )
            tx_id += 1

    df = pd.DataFrame(records)
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    df = simulate_transactions(n_customers=4000, avg_tx_per_customer=8)
    out_path = os.path.join(raw_dir, "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic transactions to: {out_path}")
    print(df.head())
    print("Number of customers:", df['customer_id'].nunique())


if __name__ == "__main__":
    main()
