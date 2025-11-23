🚀 Customer Segmentation with RFM Clustering
Behavioral Segmentation • Marketing Analytics • Customer Intelligence

This project builds a complete RFM-based customer segmentation system using synthetic e-commerce style data.
Customers are grouped using K-Means clustering, based on the classic RFM metrics:

Recency – how recently a customer purchased

Frequency – how often they purchase

Monetary – how much they spend

The result is a realistic, production-style segmentation pipeline used in Customer Analytics, Marketing Data Science, and Retention Teams.

🔥 Key Features

Synthetic Customer Dataset Generator (10,000+ customers)

Full RFM Pipeline

Compute R, F, M metrics

RobustScaler normalization

Automated cluster count selection (Silhouette Score)

KMeans Clustering

Segment Labeling (Champions, Loyal, At Risk, etc.)

Visualization Tools

RFM Heatmaps

3D RFM plot

Cluster distributions

Streamlit Dashboard for interactive exploration

Clean, modular, reproducible ML project structure

📂 Project Structure
customer-segmentation-rfm/
│── data/

│   └── synthetic_customers.csv

│── outputs/

│   ├── rfm_scores.csv

│   ├── cluster_assignments.csv

│   └── plots/

│── src/

│   ├── generate_data.py

│   ├── preprocess.py

│   ├── rfm_clustering.py

│   └── visualize.py

│── requirements.txt

│── README.md

└── app.py


🧪 How the Pipeline Works
1️⃣ Generate Synthetic Customer Data
python src/generate_data.py


Creates a realistic customer behavior dataset with Recency, Frequency, and Monetary values.

2️⃣ Preprocess & Calculate RFM
python src/preprocess.py


Cleans data

Removes extreme outliers

Creates R, F, M metrics

Saves processed dataset

3️⃣ Run KMeans Clustering
python src/rfm_clustering.py


Determines optimal cluster count using Silhouette Score

Performs KMeans clustering

Assigns business-friendly segment names

4️⃣ Launch the Streamlit Dashboard
streamlit run app.py


The dashboard includes:

Segment distribution visualization

RFM heatmaps

3D RFM plots

Customer-level exploration

📊 Example Segments
Segment	Description	Business Meaning
Champions	High Recency, Frequency, Monetary	Most valuable customers
Loyal Customers	Frequent buyers	Strong relationship with brand
At Risk	High Monetary but older Recency	Ideal for win-back campaigns
Hibernating	Low R/F/M	Inactive and low engagement
New Customers	Recently joined	Best for onboarding campaigns
🧠 Business Use Cases

✔ Retention & churn prevention strategies
✔ Personalized marketing & lifecycle campaigns
✔ Customer Lifetime Value (CLV) enhancement
✔ Product analytics segmentation
✔ Behavior-driven recommendations
✔ Growth & revenue optimization

⚙️ Installation

git clone https://github.com/abcanli/customer-segmentation-rfm-clustering.git
cd customer-segmentation-rfm-clustering
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

👤 Author

Ali Berk Canlı
NLP/ML Analyst • Data / Product Analytics

GitHub: https://github.com/abcanli

LinkedIn: https://www.linkedin.com/in/aliberkcanlı



