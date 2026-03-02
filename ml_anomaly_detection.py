import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import numpy as np

import os
from dotenv import load_dotenv
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "goszakup_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "0000")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def run_ml_anomaly_detection():
    print("Starting ML Anomaly Detection (Isolation Forest)...")
    engine = create_engine(DB_URI)
    
    # Extract historical core data
    query = """
    SELECT 
        c.id as lot_id,
        p.number_anno,
        p.org_bin,
        p.org_name_ru,
        c.clean_name,
        COALESCE(c.enstru_code, 'UNKNOWN') as enstru_code,
        c.unit_price,
        c.count as quantity,
        c.amount as total_amount,
        c.purchase_year
    FROM core_lots_cleaned c
    JOIN purchases p ON c.purchase_id = p.id
    WHERE c.unit_price > 0 AND c.count > 0;
    """
    
    print("Fetching data from core_lots_cleaned...")
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("No data available for ML training.")
        return
        
    print(f"Data fetched successfully. Rows: {len(df)}")
    
    # We will detect multidimensional anomalies within each ENSTRU category per year, 
    # to find purchases that behave suspiciously in terms of both PRICE and VOLUME logic.
    # Grouping by ENSTRU and Year to analyze comparable contexts
    
    # To keep the model robust, we only run IF on categories that have > 10 transactions
    counts = df.groupby(['enstru_code', 'purchase_year']).size().reset_index(name='trx_count')
    valid_groups = counts[counts['trx_count'] > 10]
    
    df_valid = df.merge(valid_groups[['enstru_code', 'purchase_year']], on=['enstru_code', 'purchase_year'])
    print(f"Applying Isolation Forest on {len(df_valid)} qualifying rows...")
    
    anomalies_list = []
    
    # Processing each group independently 
    groups = df_valid.groupby(['enstru_code', 'purchase_year'])
    for (enstru, year), group in groups:
        # Features for Isolation Forest: we log transform strictly positive features 
        # because purchase volume distributions are usually highly skewed
        features = np.log1p(group[['unit_price', 'quantity']].values)
        
        # Isolation Forest configuration
        iso = IsolationForest(contamination=0.05, random_state=42) # Expecting ~5% anomalies
        
        # Fit and Predict (-1 for outliers, 1 for inliers)
        preds = iso.fit_predict(features)
        
        # Filter only anomalies
        anomalous_rows = group[preds == -1].copy()
        
        if not anomalous_rows.empty:
            # We calculate an anomaly score representing how isolated this point is inside the tree
            scores = iso.score_samples(features)
            anomalous_rows['ml_score'] = scores[preds == -1]
            anomalies_list.append(anomalous_rows)

    if not anomalies_list:
        print("No ML anomalies detected.")
        return
        
    final_anomalies_df = pd.concat(anomalies_list, ignore_index=True)
    print(f"Detected {len(final_anomalies_df)} multi-dimensional anomalies.")
    
    # Sort by the most severe anomaly scores first (lower is more anomalous)
    final_anomalies_df = final_anomalies_df.sort_values(by='ml_score')
    
    from sqlalchemy import text
    
    # Drop table and write back to database
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS mart_ml_anomalies;"))
    
    print("Writing results to mart_ml_anomalies table...")
    final_anomalies_df.to_sql('mart_ml_anomalies', engine, if_exists='replace', index=False)
    
    # Create an index for faster agent lookups
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX idx_ml_anomalies_name ON mart_ml_anomalies(clean_name);"))
        conn.execute(text("CREATE INDEX idx_ml_anomalies_org ON mart_ml_anomalies(org_name_ru);"))
        
    print("ML Anomaly Detection Pipeline completed successfully!")

if __name__ == "__main__":
    run_ml_anomaly_detection()
