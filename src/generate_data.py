import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def generate_synthetic_ecommerce(n_customers: int = 1200):
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

    rows = []
    today = datetime(2024, 6, 30)

    for cust_id in customer_ids:
        # Sipariş sayısı (Poisson)
        n_orders = np.random.poisson(lam=5)
        n_orders = max(n_orders, 1)

        # Sipariş tarihleri (son 365 gün)
        order_dates = [
            today - timedelta(days=int(np.random.uniform(0, 365)))
            for _ in range(n_orders)
        ]
        order_dates = sorted(order_dates)

        # Harcama baz seviyesi
        base_spend = np.random.choice(
            [30, 50, 80, 120, 200],
            p=[0.25, 0.30, 0.25, 0.15, 0.05],
        )

        for i, od in enumerate(order_dates):
            amount = np.round(
                np.random.gamma(shape=2.0, scale=base_spend / 2.0)
                + np.random.uniform(5, 40),
                2,
            )

            rows.append(
                {
                    "customer_id": cust_id,
                    "order_id": f"{cust_id}_ORD_{i+1:03d}",
                    "order_date": od.date().isoformat(),
                    "amount": float(amount),
                }
            )

    return pd.DataFrame(rows)


def main():
    df = generate_synthetic_ecommerce()
    out_path = os.path.join(RAW_DIR, "ecommerce_synthetic.csv")

    df.to_csv(out_path, index=False)
    print(f"✅ Synthetic dataset saved → {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
