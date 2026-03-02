import requests
import psycopg2
from psycopg2 import extras
import time
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# 1. КОНФИГУРАЦИЯ ПОДКЛЮЧЕНИЯ
# Исправляем возможные проблемы с кодировкой в Windows
if sys.platform == 'win32':
    os.environ['PGCLIENTENCODING'] = 'utf8'

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "goszakup_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "options": "-c client_encoding=utf8"
}

TOKEN = os.getenv("GOSZAKUP_API_TOKEN")
URL = "https://ows.goszakup.gov.kz/v3/graphql"

# Список БИН из твоего ТЗ
BINS = [
'000740001307', '020240002363', '020440003656', '030440003698',
'050740004819','051040005150', '100140011059', '120940001946', '140340016539',
'150540000186','171041003124', '210240019348', '210240033968', '210941010761',
'230740013340','231040023028', '780140000023', '900640000128', '940740000911',
'940940000384','960440000220', '970940001378', '971040001050', '980440001034',
'981140001551','990340005977', '990740002243'
]


def init_db():
    """Инициализация таблиц согласно ТЗ"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # Таблица объявлений
    cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id BIGINT PRIMARY KEY,
            number_anno TEXT,
            name_ru TEXT,
            total_sum NUMERIC,
            fin_year INT,
            org_bin TEXT,
            org_name_ru TEXT,
            publish_date TIMESTAMP,
            kato_code TEXT,
            status_id INT,
            trade_method_id INT
        );
    """)
    # Таблица лотов с полем unit_price для анализа Fair Price
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lots (
            id BIGINT PRIMARY KEY,
            purchase_id BIGINT REFERENCES purchases(id),
            lot_number TEXT,
            name_ru TEXT,
            count NUMERIC,
            amount NUMERIC,
            unit_price NUMERIC,
            enstru_list TEXT,
            unit_ru TEXT,
            winner_bin TEXT,
            winner_name TEXT
        );
    """)
    # Таблица договоров (Contract API)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id BIGINT PRIMARY KEY,
            number_contract TEXT,
            sign_date TIMESTAMP,
            contract_sum NUMERIC,
            supplier_bin TEXT,
            supplier_name_ru TEXT,
            customer_bin TEXT,
            customer_name_ru TEXT,
            trd_buy_id BIGINT,
            lot_id BIGINT,
            contract_year INT,
            status_id INT,
            item_name TEXT,
            unit_price NUMERIC,
            quantity NUMERIC
        );
    """)
    conn.commit()
    return conn

def get_last_publish_date(conn):
    """Возвращает дату последней загруженной публикации для инкрементальной загрузки."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(publish_date) FROM purchases")
    res = cur.fetchone()
    return res[0] if res and res[0] else None

def fetch_page_with_retry(bin_code, after=None, publish_date_from=None, retries=5):
    """Запрос к API с механизмом повторов при обрыве соединения"""
    # Build query vars and filter dynamically to avoid 'unused variable' GraphQL errors
    if publish_date_from:
        query_vars = "$bin: String, $after: Int, $publishDate: [String]"
        filter_fields = "orgBin: $bin, publishDate: $publishDate"
    else:
        query_vars = "$bin: String, $after: Int"
        filter_fields = "orgBin: $bin"

    query = f"""
    query getPurchases({query_vars}) {{
      TrdBuy(filter: {{{filter_fields}}}, limit: 50, after: $after) {{
        id
        numberAnno
        nameRu
        totalSum
        finYear
        orgBin
        orgNameRu
        publishDate
        kato
        refBuyStatusId
        refTradeMethodsId
        biinSupplier
        Lots {{
          id
          lotNumber
          nameRu
          count
          amount
          enstruList
        }}
      }}
    }}
    """
    variables = {"bin": bin_code, "after": after}
    if publish_date_from:
        variables["publishDate"] = [publish_date_from.strftime("%Y-%m-%d %H:%M:%S")]
    
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    
    for attempt in range(retries):
        try:
            response = requests.post(URL, json={'query': query, 'variables': variables}, headers=headers, timeout=40)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, http.client.RemoteDisconnected) as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 3
                print(f"  [!] Ошибка сети (БИН {bin_code}). Попытка {attempt+1}/{retries}. Ждем {wait}с...")
                time.sleep(wait)
            else:
                print(f"  [X] Не удалось загрузить данные после {retries} попыток.")
                return None

import http.client # Нужно для отлова RemoteDisconnected

def main():
    import sys
    full_reload = "--full-reload" in sys.argv

    conn = init_db()
    cur = conn.cursor()

    if full_reload:
        print("[*] ПОЛНАЯ ПЕРЕЗАГРУЗКА: пересохраняем все записи (kato, unit_ru, winners будут обновлены).")
        last_date = None
    else:
        last_date = get_last_publish_date(conn)
        if last_date:
            print(f"[*] Инкрементальная загрузка: ищем записи новее {last_date}")

    for bin_code in BINS:
        print(f"\n--- Начинаем загрузку БИН: {bin_code} ---")
        last_id = None
        has_next = True
        total_saved = 0
        
        while has_next:
            result = fetch_page_with_retry(bin_code, last_id, publish_date_from=last_date)
            
            if not result or "data" not in result or result.get("data") is None:
                print(f"Ошибка или пустые данные для БИН {bin_code}")
                break
                
            data = result["data"].get("TrdBuy", [])
            page_info = result.get("extensions", {}).get("pageInfo", {})
            
            if not data:
                break

            for p in data:
                # Трансформация данных: извлекаем год из списка
                f_year = p['finYear'][0] if p['finYear'] and len(p['finYear']) > 0 else None
                
                # Запись объявления
                cur.execute("""
                    INSERT INTO purchases (id, number_anno, name_ru, total_sum, fin_year, org_bin, org_name_ru, publish_date, kato_code, status_id, trade_method_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                    ON CONFLICT (id) DO UPDATE SET total_sum = EXCLUDED.total_sum, kato_code = EXCLUDED.kato_code, status_id = EXCLUDED.status_id
                """, (p['id'], p['numberAnno'], p['nameRu'], p['totalSum'], f_year, p['orgBin'], p['orgNameRu'], p['publishDate'], p.get('kato'), p.get('refBuyStatusId'), p.get('refTradeMethodsId')))
                
                # Запись лотов с расчетом unit_price
                lots_data = p.get('Lots')
                if lots_data:
                    for l in lots_data:
                        cnt = float(l.get('count') or 0)
                        amt = float(l.get('amount') or 0)
                        u_price = amt / cnt if cnt > 0 else 0
                        
                        cur.execute("""
                            INSERT INTO lots (id, purchase_id, lot_number, name_ru, count, amount, unit_price, enstru_list, unit_ru, winner_bin, winner_name)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                            ON CONFLICT (id) DO UPDATE SET winner_bin = EXCLUDED.winner_bin, winner_name = EXCLUDED.winner_name
                        """, (l['id'], p['id'], l['lotNumber'], l['nameRu'], cnt, amt, u_price, str(l['enstruList']), l.get('unitNameRu'), l.get('winnerBin'), l.get('winnerNameRu')))
                else:
                    print(f"  [!] Объявление {p['id']} не содержит лотов. Пропускаем.")
                    
            conn.commit()
            total_saved += len(data)
            
            has_next = page_info.get("hasNextPage", False)
            last_id = page_info.get("lastId")
            
            print(f"  Загружено {len(data)} (Всего: {total_saved}). Next ID: {last_id}")
            time.sleep(0.5) # Пауза для стабильности API

    conn.close()
    print("\n✅ Загрузка объявлений и лотов завершена!")
    
    # Step 2: Load contracts
    print("\n--- Начинаем загрузку договоров (Contract API) ---")
    from load_contracts import load_contracts
    load_contracts()

if __name__ == "__main__":
    main()
