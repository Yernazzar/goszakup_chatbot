from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import time

import os
from dotenv import load_dotenv
load_dotenv()

# Список БИН из ТЗ
BINS = [
    '000740001307', '020240002363', '020440003656', '030440003698',
    '050740004819', '051040005150', '100140011059', '120940001946', '140340016539',
    '150540000186', '171041003124', '210240019348', '210240033968', '210941010761',
    '230740013340', '231040023028', '780140000023', '900640000128', '940740000911',
    '940940000384', '960440000220', '970940001378', '971040001050', '980440001034',
    '981140001551', '990340005977', '990740002243'
]

TOKEN = os.getenv("GOSZAKUP_API_TOKEN", "9e3c82c2e1c542588ef7ae4484e073b4")
URL = "https://ows.goszakup.gov.kz/v3/graphql"

def fetch_page_with_retry(bin_code, after=None, retries=5):
    """Запрос к API с механизмом повторов при обрыве соединения"""
    query = """
    query getPurchases($bin: String, $after: Int) {
      TrdBuy(filter: {orgBin: $bin}, limit: 50, after: $after) {
        id
        numberAnno
        nameRu
        totalSum
        finYear
        orgBin
        orgNameRu
        publishDate
        kato
        Lots {
          id
          lotNumber
          nameRu
          count
          amount
          enstruList
        }
      }
    }
    """
    variables = {"bin": bin_code, "after": after}
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    
    for attempt in range(retries):
        try:
            response = requests.post(URL, json={'query': query, 'variables': variables}, headers=headers, timeout=40)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                return None

def extract_and_load(**kwargs):
    # Используем PostgresHook для безопасного извлечения кредов Airflow Connection (Conn Id = goszakup_dwh)
    pg_hook = PostgresHook(postgres_conn_id='goszakup_dwh')
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bin_code in BINS:
        last_id = None
        has_next = True
        
        while has_next:
            result = fetch_page_with_retry(bin_code, last_id)
            if not result or "data" not in result or result.get("data") is None:
                break
                
            data = result["data"].get("TrdBuy", [])
            page_info = result.get("extensions", {}).get("pageInfo", {})
            
            if not data:
                break

            for p in data:
                f_year = p['finYear'][0] if p['finYear'] and len(p['finYear']) > 0 else None
                
                cur.execute("""
                    INSERT INTO stg_purchases (id, number_anno, name_ru, total_sum, fin_year, org_bin, org_name_ru, publish_date, kato_code)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
                    ON CONFLICT (id) DO UPDATE SET total_sum = EXCLUDED.total_sum, kato_code = EXCLUDED.kato_code
                """, (p['id'], p['numberAnno'], p['nameRu'], p['totalSum'], f_year, p['orgBin'], p['orgNameRu'], p['publishDate'], p.get('kato')))
                
                lots_data = p.get('Lots')
                if lots_data:
                    for l in lots_data:
                        cnt = float(l.get('count') or 0)
                        amt = float(l.get('amount') or 0)
                        u_price = amt / cnt if cnt > 0 else 0
                        
                        cur.execute("""
                            INSERT INTO stg_lots (id, purchase_id, lot_number, name_ru, count, amount, unit_price, enstru_list)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING
                        """, (l['id'], p['id'], l['lotNumber'], l['nameRu'], cnt, amt, u_price, str(l['enstruList'])))
                        
            conn.commit()
            has_next = page_info.get("hasNextPage", False)
            last_id = page_info.get("lastId")
            time.sleep(0.5)

    conn.close()

def refresh_marts(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='goszakup_dwh')
    conn = pg_hook.get_conn()
    cur = conn.cursor()
    # Обновляем view / mat view для Mart Layer
    cur.execute("REFRESH MATERIALIZED VIEW core_lots_cleaned;")
    # Если используются обычные View, то их пересоздавать не надо. 
    # В нашем случае это Views, но для Production лучше использовать dbt.
    conn.commit()
    conn.close()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'goszakup_ingestion_pipeline',
    default_args=default_args,
    description='Daily ingestion of goszakup API OWS v3',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id='extract_and_load_data',
        python_callable=extract_and_load,
    )

    refresh_marts_task = PythonOperator(
        task_id='refresh_data_marts',
        python_callable=refresh_marts,
    )

    def trigger_ml():
        from subprocess import run
        import os
        # Path to the ml script assuming it's in the same project root
        script_path = os.path.join(os.path.dirname(__file__), '..', 'ml_anomaly_detection.py')
        run(['python', script_path], check=True)

    def trigger_contracts():
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from load_contracts import load_contracts
        load_contracts()

    ingest_task = PythonOperator(
        task_id='extract_and_load_data',
        python_callable=extract_and_load,
    )

    refresh_marts_task = PythonOperator(
        task_id='refresh_data_marts',
        python_callable=refresh_marts,
    )

    ml_anomalies_task = PythonOperator(
        task_id='ml_anomaly_detection',
        python_callable=trigger_ml,
    )

    contracts_task = PythonOperator(
        task_id='load_contracts',
        python_callable=trigger_contracts,
    )

    ingest_task >> contracts_task >> refresh_marts_task >> ml_anomalies_task
