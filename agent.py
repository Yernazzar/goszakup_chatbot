import os
from dotenv import load_dotenv
load_dotenv()
import json
import psycopg2
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Import Fair Price calculation module
try:
    from fair_price import calculate_fair_price, initialize_reference_tables
except ImportError:
    print("[WARNING] fair_price module not found. Fair Price calculations disabled.")
    calculate_fair_price = None

# Lazy-loaded multilingual sentence transformer (cached after first call)
@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 1. Database Configuration ---
DB_CONFIG = {
    'dbname': os.getenv("DB_NAME", "goszakup_db"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD", "0000"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

def execute_sql(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Helper to execute SQL and return dicts."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        return [{"error": str(e)}]


# --- 2. KATO / Region Mapping ---
# Official Kazakhstan KATO: 9-digit codes; first 2 digits identify the region.
# This dict maps any user-written variant (abbreviation, city, Kazakh spelling, etc.)
# to the 2-digit KATO prefix used in ref_regional_coefficients.kato_code.
#
# Official mapping (from https://data.egov.kz):
#  10 → Абайская область
#  11 → Акмолинская область
#  15 → Актюбинская область
#  19 → Алматинская область
#  23 → Атырауская область
#  27 → Западно-Казахстанская область
#  31 → Жамбылская область
#  33 → область Жетісу
#  35 → Карагандинская область
#  39 → Костанайская область
#  43 → Кызылординская область
#  47 → Мангистауская область
#  51 → Южно-Казахстанская область
#  55 → Павлодарская область
#  59 → Северо-Казахстанская область
#  61 → Туркестанская область
#  62 → область Ұлытау
#  63 → Восточно-Казахстанская область
#  71 → г.Астана
#  75 → г.Алматы
#  79 → г.Шымкент
KATO_ALIASES: Dict[str, str] = {
    # Абайская область  → '10'
    'абай': '10', 'абайская': '10', 'абайская область': '10',
    'abai': '10', 'abay': '10',

    # Акмолинская область  → '11'
    'акмола': '11', 'акмолинская': '11', 'акмолинская область': '11',
    'кокшетау': '11', 'akmola': '11', 'kokshetau': '11',

    # Актюбинская область  → '15'
    'актобе': '15', 'актюбе': '15', 'актюбинская': '15', 'актюбинская область': '15',
    'aktobe': '15', 'aqtobe': '15',

    # Алматинская область  → '19'
    'алматинская': '19', 'алматинская область': '19',
    'almaty oblast': '19', 'almatinskaya': '19',

    # Атырауская область  → '23'
    'атырау': '23', 'атырауская': '23', 'атырауская область': '23',
    'atyrau': '23', 'atyrau oblast': '23',

    # Западно-Казахстанская область  → '27'
    'зко': '27', 'западно-казахстанская': '27', 'западно-казахстанская область': '27',
    'уральск': '27', 'западный': '27', 'zko': '27', 'uralsk': '27',

    # Жамбылская область  → '31'
    'жамбыл': '31', 'жамбылская': '31', 'жамбылская область': '31',
    'тараз': '31', 'zhambyl': '31', 'jambyl': '31', 'taraz': '31',

    # область Жетісу  → '33'
    'жетісу': '33', 'жетису': '33', 'область жетісу': '33',
    'zhetysu': '33', 'jetysu': '33',

    # Карагандинская область  → '35'
    'караганда': '35', 'карагандинская': '35', 'карагандинская область': '35',
    'қарағанды': '35', 'karaganda': '35', 'karagandy': '35', 'qaraghandy': '35',

    # Костанайская область  → '39'
    'костанай': '39', 'кустанай': '39', 'костанайская': '39', 'костанайская область': '39',
    'kostanay': '39', 'qostanai': '39',

    # Кызылординская область  → '43'
    'кызылорда': '43', 'кызылординская': '43', 'кызылординская область': '43',
    'kyzylorda': '43', 'qyzylorda': '43',

    # Мангистауская область  → '47'
    'мангистау': '47', 'мангышлак': '47', 'мангистауская': '47', 'мангистауская область': '47',
    'актау': '47', 'mangystau': '47', 'aktau': '47',

    # Южно-Казахстанская область  → '51'
    'юко': '51', 'южно-казахстанская': '51', 'южно-казахстанская область': '51',
    'южный казахстан': '51', 'yko': '51',

    # Павлодарская область  → '55'
    'павлодар': '55', 'павлодарская': '55', 'павлодарская область': '55',
    'pavlodar': '55',

    # Северо-Казахстанская область  → '59'
    'ско': '59', 'северо-казахстанская': '59', 'северо-казахстанская область': '59',
    'петропавловск': '59', 'северный казахстан': '59', 'sko': '59', 'petropavlovsk': '59',

    # Туркестанская область  → '61'
    'туркестан': '61', 'түркістан': '61', 'туркестанская': '61', 'туркестанская область': '61',
    'turkestan': '61', 'turkistan': '61',

    # область Ұлытау  → '62'
    'улытау': '62', 'Ұлытау': '62', 'область Ұлытау': '62',
    'ulytau': '62',

    # Восточно-Казахстанская область  → '63'
    'вко': '63', 'восточно-казахстанская': '63', 'восточно-казахстанская область': '63',
    'восточный казахстан': '63', 'усть-каменогорск': '63',
    'vko': '63', 'oskemen': '63', 'ust-kamenogorsk': '63',

    # г.Астана  → '71'
    'астана': '71', 'нур-султан': '71', 'нурсултан': '71',
    'astana': '71', 'nur-sultan': '71', 'nursultan': '71',

    # г.Алматы  → '75'
    'алматы': '75', 'алма-ата': '75', 'алмата': '75',
    'almaty': '75', 'alma-ata': '75',

    # г.Шымкент  → '79'
    'шымкент': '79', 'шимкент': '79', 'shymkent': '79', 'shimkent': '79',
}


def normalize_region(user_input: str) -> str:
    """
    Convert any user-written region name/abbreviation/city to the 2-digit KATO
    prefix used in ref_regional_coefficients.kato_code.

    Returns the 2-digit prefix string if found (e.g. '63' for 'ВКО'),
    or None if no match — so the caller can decide whether to filter or not.
    """
    key = user_input.strip().lower()
    return KATO_ALIASES.get(key, None)


# --- 3. Langchain Tools Definition (Data Fetching) ---

@tool
def get_anomalies_by_product(product_name: str = "", enstru_code: str = "",
                              region_name: str = "", year: int = 0,
                              min_deviation_percent: float = 30.0, limit: int = 5) -> str:
    """
    Search for purchases (lots) where the price exceeds the fair market price (deviation > threshold).
    PRIMARY use case: Class 1 query — 'Find purchases with price deviation > 30% from weighted-average
    price for a given ENSTRU code'.

    Parameters:
    - product_name: partial product name search (optional if enstru_code is provided)
    - enstru_code:  ENSTRU/PKNI category code for the product (e.g. '36101500'). If provided,
                   filters EXACTLY by ENSTRU code — this is the correct approach per the spec.
    - region_name:  any region form ('ВКО', 'Астана', 'Тараз', 'Мангистауская' etc.)
    - year:         purchase year (e.g. 2024)
    - min_deviation_percent: minimum % overprice vs fair price (default 30)
    """
    conditions = ["deviation_percent > %s", "fair_price > 0"]
    params = [min_deviation_percent]

    if enstru_code:
        # Exact match by official product category code — preferred over name search
        conditions.append("enstru_code = %s")
        params.append(enstru_code.strip())
    elif product_name:
        conditions.append("clean_name ILIKE %s")
        params.append(f"%{product_name.lower().strip()}%")
    else:
        return "Укажите product_name или enstru_code для поиска."

    if region_name:
        kato_prefix = normalize_region(region_name)
        if kato_prefix:
            conditions.append("LEFT(kato_code::TEXT, 2) = %s")
            params.append(kato_prefix)
        else:
            conditions.append("region_name ILIKE %s")
            params.append(f"%{region_name.strip()}%")
    if year > 0:
        conditions.append("purchase_year = %s")
        params.append(year)

    query = f'''
        SELECT
            org_bin, org_name_ru, clean_name, enstru_code, purchase_year,
            region_name, unit_price, fair_price, baseline_price,
            deviation_percent, sample_count, number_anno
        FROM mart_fair_price
        WHERE {" AND ".join(conditions)}
        ORDER BY deviation_percent DESC
        LIMIT %s
    '''
    params.append(limit)
    data = execute_sql(query, tuple(params))

    label = enstru_code or product_name
    if not data or "error" in data[0]:
        return f"Данные по '{label}' с отклонением >{min_deviation_percent}% не найдены."

    lines = [f"Анализ ценовых аномалий: {label}\n"]
    lines.append(f"Выявлено {len(data)} закупок с отклонением от справедливой цены выше {min_deviation_percent}%.")
    if data[0].get('enstru_code'):
        lines.append(f"ЕНСТРУ-код: {data[0]['enstru_code']}")
    lines.append("")

    for idx, row in enumerate(data, 1):
        actual = float(row['unit_price'])
        fair = float(row['fair_price'])
        dev = float(row['deviation_percent'])
        dev_str = f"в {round(dev/100 + 1, 1)} раз" if dev > 1000 else f"{round(dev, 1)}%"

        lines.append(f"{idx}. Заказчик: {row['org_name_ru']} (БИН: {row.get('org_bin', 'Н/Д')})")
        lines.append(f"   Товар: {row['clean_name']} ({row['purchase_year']} г., {row['region_name'] or 'РК'})")
        lines.append(f"   Цена: {actual:,.0f} тг (справедливая: {fair:,.0f} тг)")
        lines.append(f"   Превышение: {dev_str} | Выборка: {row.get('sample_count', '?')} аналогов")
        lines.append(f"   [Ссылка на портал](https://goszakup.gov.kz/ru/announce/index/{row['number_anno']})")
        lines.append("")

    return "\n".join(lines)


@tool
def evaluate_org_fairness(org_name: str) -> str:
    """
    Evaluates how fair and efficient an organization (by name or partial name) conducts its purchases.
    Analyzes anomalies with Fair Price metric considering regional and temporal factors.
    Use this tool when the user asks about a specific organization's rating, toxicity, or procurement fairness.
    """
    query = '''
        SELECT 
            org_bin, org_name_ru,
            COUNT(*) as total_lots,
            COUNT(CASE WHEN deviation_percent > 30 THEN 1 END) as anomalous_lots,
            ROUND(AVG(deviation_percent)::NUMERIC, 2) as avg_deviation_percent,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY deviation_percent)::NUMERIC, 2) as median_deviation_percent,
            ROUND(MAX(deviation_percent)::NUMERIC, 2) as max_deviation_percent,
            ROUND(SUM(unit_price - fair_price)::NUMERIC, 0) as total_overpay
        FROM mart_fair_price
        WHERE org_name_ru ILIKE %s AND fair_price > 0
        GROUP BY org_bin, org_name_ru
        LIMIT 3
    '''
    search_term = f"%{org_name.strip()}%"
    data = execute_sql(query, (search_term,))
    
    if not data or "error" in data[0]:
        return f"❌ Организация '{org_name}' не найдена в аналитике закупок."

    lines = []
    for row in data:
        org = row['org_name_ru']
        bin_code = row['org_bin']
        total = row['total_lots']
        anomalous = row['anomalous_lots']
        avg_dev = row['avg_deviation_percent']
        median_dev = row['median_deviation_percent']
        max_dev = row['max_deviation_percent']
        total_overpay = row['total_overpay'] or 0
        
        # Calculate fairness rating
        if total > 0:
            anomaly_rate = (anomalous / total) * 100
            if anomaly_rate > 50:
                rating = "🔴 КРИТИЧЕСКАЯ (>50% аномалий)"
            elif anomaly_rate > 30:
                rating = "🟠 ВЫСОКАЯ (30-50% аномалий)"
            elif anomaly_rate > 10:
                rating = "🟡 СРЕДНЯЯ (10-30% аномалий)"
            else:
                rating = "🟢 НИЗКАЯ (<10% аномалий)"
        else:
            rating = "⚪ НЕТ ДАННЫХ"
        
    lines.append(f"📋 {org} (БИН: {bin_code})\n")
    if total > 0:
        lines.append(f"Рейтинг справедливости: {rating}\n")
        lines.append(f"Статистика:")
        lines.append(f"  • Всего закупок: {total}")
        lines.append(f"  • С завышением цены (>30%): {anomalous}")
        lines.append(f"  • Среднее завышение: {round(avg_dev, 1)}%")
        lines.append(f"  • Максимальное завышение: {round(max_dev, 1)}%")
        lines.append(f"  • Общая переплата: {total_overpay:,.0f} тг\n")

        # Fetch top anomalies for this org if any
        if anomalous > 0:
            top_anomalies_query = '''
                SELECT 
                    number_anno, clean_name, purchase_year, region_name,
                    unit_price, fair_price, deviation_percent
                FROM mart_fair_price
                WHERE org_bin = %s AND deviation_percent > 30 AND fair_price > 0
                ORDER BY deviation_percent DESC
                LIMIT 3
            '''
            top_data = execute_sql(top_anomalies_query, (bin_code,))
            if top_data and "error" not in top_data[0]:
                lines.append("Топ подозрительных закупок организации:")
                for idx, r in enumerate(top_data, 1):
                    actual = float(r['unit_price'])
                    fair = float(r['fair_price'])
                    dev = float(r['deviation_percent'])
                    
                    if dev > 1000:
                        dev_str = f"в {round(dev/100 + 1, 1)} раз"
                    else:
                        dev_str = f"{round(dev, 1)}%"

                    lines.append(f"  {idx}. Товар: {r['clean_name']}")
                    lines.append(f"     Цена: {actual:,.0f} тг (справедливая: {fair:,.0f} тг)")
                    lines.append(f"     Превышение: {dev_str}")
                    lines.append(f"     [Ссылка](https://goszakup.gov.kz/ru/announce/index/{r['number_anno']})")
                lines.append("")
    
    return "\n".join(lines)


@tool
def evaluate_lot_fairness(number_anno: str) -> str:
    """
    CLASS 2 TOOL: Evaluate the price fairness of a SPECIFIC LOT (by announcement number).
    Compares the lot's unit price against similar contracts (same ENSTRU code) from
    OTHER organisations in the SAME delivery city/region (by KATO prefix).

    Use this when the user asks:
    - 'Assess lot №...'
    - 'Is the price of announcement №... adequate?'
    - 'Compare lot №... against similar contracts in the same city'
    """
    # Step 1: fetch the target lot
    lot_query = """
        SELECT
            m.lot_id, m.number_anno, m.org_bin, m.org_name_ru, m.clean_name, m.enstru_code,
            m.kato_code, m.region_name, m.unit_price, m.fair_price, m.baseline_price,
            m.deviation_percent, m.sample_count, m.purchase_year, m.purchase_month
        FROM mart_fair_price m
        LEFT JOIN lots l ON m.lot_id = l.id
        WHERE m.number_anno::TEXT = %s OR l.lot_number::TEXT = %s OR m.lot_id::TEXT = %s
        LIMIT 1
    """
    param = str(number_anno).strip()
    lot = execute_sql(lot_query, (param, param, param))
    if not lot or "error" in lot[0]:
        return (f"Лот с номером объявления '{number_anno}' не найден в базе. "
                "Убедитесь, что номер соответствует полю number_anno на портале госзакупок.")

    row = lot[0]
    enstru = row.get('enstru_code') or ''
    kato = row.get('kato_code') or ''
    kato_prefix = str(kato)[:2] if kato else None
    year = row.get('purchase_year') or 0
    org_bin = row.get('org_bin') or ''
    unit_price = float(row.get('unit_price') or 0)
    fair_price = float(row.get('fair_price') or 0)
    dev = float(row.get('deviation_percent') or 0)

    lines = [f"Оценка справедливости цены: Лот (объявление №{number_anno})\n"]
    lines.append(f"Заказчик:    {row['org_name_ru']} (БИН: {org_bin})")
    enstru_txt = f" (ЕНСТРУ: {enstru})" if enstru and enstru != 'UNKNOWN' else ""
    lines.append(f"Товар:       {row['clean_name']}{enstru_txt}")
    lines.append(f"Регион:      {row['region_name'] or 'Н/Д'} | Год: {year}")
    lines.append(f"Цена:        {unit_price:,.0f} тг")
    if fair_price > 0:
        dev_str = f"в {round(dev/100+1,1)} раз ВЫШЕ" if dev > 1000 else f"{round(dev,1)}%"
        verdict = "ЗАВЫШЕНА" if dev > 30 else ("ЗАНИЖЕНА" if dev < -20 else "АДЕКВАТНА")
        lines.append(f"Справедливая цена: {fair_price:,.0f} тг (отклонение: {dev_str})")
        lines.append(f"Вердикт цены: {verdict}")
        lines.append(f"Размер выборки: {row.get('sample_count', 0)} аналогов")
    lines.append("")

    # Step 2: compare with OTHER orgs in the same KATO + same ENSTRU
    if enstru and enstru != 'UNKNOWN' and kato_prefix:
        peers_query = """
            SELECT
                org_name_ru, unit_price, purchase_year, deviation_percent, number_anno
            FROM mart_fair_price
            WHERE enstru_code = %s
              AND LEFT(kato_code::TEXT, 2) = %s
              AND org_bin != %s
              AND fair_price > 0
            ORDER BY ABS(purchase_year - %s) ASC, deviation_percent DESC
            LIMIT 5
        """
        peers = execute_sql(peers_query, (enstru, kato_prefix, org_bin, year))
        if peers and "error" not in peers[0]:
            prices = [float(p['unit_price']) for p in peers]
            median_peer = sorted(prices)[len(prices)//2]
            diff_pct = ((unit_price - median_peer) / median_peer * 100) if median_peer > 0 else 0

            lines.append(f"Сравнение с аналогами других ведомств ({len(peers)} шт., тот же город/КАТО):")
            lines.append(f"  Медианная цена аналогов: {median_peer:,.0f} тг")
            lines.append(f"  Отклонение от медианы:   {round(diff_pct, 1)}%")
            lines.append("")
            lines.append("  Топ аналогичных закупок:")
            for p in peers:
                p_dev = float(p['deviation_percent'] or 0)
                lines.append(f"  - {p['org_name_ru']}: {float(p['unit_price']):,.0f} тг "
                             f"({p['purchase_year']} г., отклонение {round(p_dev,1)}%)")
                lines.append(f"    [Объявление](https://goszakup.gov.kz/ru/announce/index/{p['number_anno']})")
        else:
            lines.append("Аналогичные закупки того же КАТО/ЕНСТРУ у других ведомств не найдены.")

    return "\n".join(lines)


@tool
def get_volume_anomalies(product_name: str = "", enstru_code: str = "",
                         org_name: str = "", limit: int = 10) -> str:
    """
    CLASS 3 TOOL: Detect UNUSUAL QUANTITY (volume) increases year-over-year.
    Compares the quantity purchased for each product per organisation across years
    and flags cases where the quantity is statistically abnormal (Z-score > 2).

    Use this when the user asks:
    - 'Find unusual quantity increases compared to previous years'
    - 'Detect atypical volume growth for product X'
    - 'Which organisations buy abnormally large quantities of Y?'
    """
    conditions = ["quantity > 0"]
    params: list = []

    if enstru_code:
        conditions.append("enstru_code = %s")
        params.append(enstru_code.strip())
    elif product_name:
        conditions.append("clean_name ILIKE %s")
        params.append(f"%{product_name.lower().strip()}%")
    else:
        return "Укажите product_name или enstru_code для анализа объёмов."

    if org_name:
        conditions.append("org_name_ru ILIKE %s")
        params.append(f"%{org_name.strip()}%")

    where_clause = " AND ".join(conditions)

    # Year-over-year quantity per org: compute avg and stddev across years,
    # then flag rows where the quantity exceeds avg + 2*stddev
    query = f"""
        WITH yearly AS (
            SELECT
                org_bin,
                org_name_ru,
                clean_name,
                enstru_code,
                purchase_year,
                SUM(quantity) AS total_qty,
                COUNT(*) AS lot_count,
                SUM(unit_price * quantity) AS total_spend,
                region_name
            FROM mart_fair_price
            WHERE {where_clause}
            GROUP BY org_bin, org_name_ru, clean_name, enstru_code, purchase_year, region_name
        ),
        stats AS (
            SELECT
                org_bin,
                clean_name,
                AVG(total_qty) AS avg_qty,
                STDDEV(total_qty) AS std_qty
            FROM yearly
            GROUP BY org_bin, clean_name
            HAVING COUNT(*) >= 2  -- need at least 2 years to compare
        )
        SELECT
            y.org_name_ru,
            y.org_bin,
            y.clean_name,
            y.enstru_code,
            y.purchase_year,
            y.total_qty,
            y.lot_count,
            y.total_spend,
            y.region_name,
            s.avg_qty,
            s.std_qty,
            CASE WHEN s.std_qty > 0
                 THEN ROUND(((y.total_qty - s.avg_qty) / s.std_qty)::NUMERIC, 2)
                 ELSE 0
            END AS z_score
        FROM yearly y
        JOIN stats s ON y.org_bin = s.org_bin AND y.clean_name = s.clean_name
        WHERE s.std_qty > 0
          AND (y.total_qty - s.avg_qty) > 2 * s.std_qty  -- z-score > 2, i.e. abnormally high
        ORDER BY z_score DESC
        LIMIT %s
    """
    params.append(limit)
    data = execute_sql(query, tuple(params))

    label = enstru_code or product_name
    if not data or "error" in data[0]:
        return (f"Нетипичных объёмных аномалий по '{label}' не найдено. "
                "Это может означать, что данных по предыдущим годам недостаточно для сравнения.")

    lines = [f"Анализ аномальных объёмов закупок: {label}\n"]
    lines.append(f"Выявлено {len(data)} случаев нетипичного завышения количества (Z-score > 2).\n")

    for idx, row in enumerate(data, 1):
        avg = float(row['avg_qty'] or 0)
        actual_qty = float(row['total_qty'] or 0)
        z = float(row['z_score'] or 0)
        excess_pct = round((actual_qty / avg - 1) * 100, 1) if avg > 0 else 0

        lines.append(f"{idx}. {row['org_name_ru']} (БИН: {row['org_bin']})")
        lines.append(f"   Товар: {row['clean_name']} | ЕНСТРУ: {row.get('enstru_code','—')}")
        lines.append(f"   Регион: {row.get('region_name','Н/Д')} | Год: {row['purchase_year']}")
        lines.append(f"   Объём: {actual_qty:,.0f} ед. (средний по годам: {avg:,.1f} ед.)")
        lines.append(f"   Превышение: +{excess_pct}% от нормы | Z-score: {z}")
        lines.append(f"   Лотов в году: {row['lot_count']} | Сумма: {float(row['total_spend'] or 0):,.0f} тг")
        lines.append("")

    return "\n".join(lines)
@tool
def get_ml_anomalies(product_name: str = "", org_name: str = "", limit: int = 10) -> str:
    """
    Search for COMPLEX multi-dimensional anomalies detected by the Isolation Forest ML model.
    Combines ML detection with Fair Price analysis for comprehensive anomaly assessment.
    Use this when the user asks about suspicious procurement of VOLUME (quantity), or when 
    normal-price but suspiciously-large-quantity lots need to be found.
    """
    conditions = ["1=1"]
    params = []
    if org_name:
        conditions.append("ml.org_name_ru ILIKE %s")
        params.append(f"%{org_name.strip()}%")
    if product_name:
        conditions.append("ml.clean_name ILIKE %s")
        params.append(f"%{product_name.lower().strip()}%")
    params.append(limit)

    query = f"""
        SELECT 
            ml.number_anno, ml.org_name_ru, ml.org_bin, ml.clean_name, ml.enstru_code,
            ml.purchase_year, ml.unit_price, ml.quantity, ml.total_amount, ml.ml_score,
            fp.fair_price, fp.deviation_percent, fp.sample_count
        FROM mart_ml_anomalies ml
        LEFT JOIN mart_fair_price fp ON ml.lot_id = fp.lot_id
        WHERE {' AND '.join(conditions)}
        ORDER BY ml.ml_score ASC
        LIMIT %s
    """
    data = execute_sql(query, tuple(params))

    if not data or "error" in data[0]:
        return "❌ Многомерные ML-аномалии по вашему запросу не найдены."

    lines = ["Анализ многомерных аномалий (объем и комбинации параметров)\n"]
    lines.append(f"Система Isolation Forest выявила {len(data)} подозрительных закупок\n")
    
    for idx, row in enumerate(data, 1):
        fair_price = row.get('fair_price', 0)
        actual = float(row['unit_price'])
        
        lines.append(f"{idx}. Заказчик: {row['org_name_ru']}")
        lines.append(f"   Товар: {row['clean_name']} ({row['purchase_year']} г.)")
        lines.append(f"   Детали: {actual:,.0f} тг × {row['quantity']} ед. = {row['total_amount']:,.0f} тг")
        
        if fair_price and fair_price > 0:
            dev = row.get('deviation_percent', 0)
            lines.append(f"   Справедливая цена: {float(fair_price):,.0f} тг (отклонение: {round(dev, 1)}%)")
        
        lines.append(f"   Уровень аномальности: {round(row['ml_score'], 4)}")
        lines.append(f"   [Ссылка на портал](https://goszakup.gov.kz/ru/announce/index/{row['number_anno']})")
        lines.append("")
    
    return "\n".join(lines)



@tool
def semantic_lot_search(query: str, limit: int = 5) -> str:
    """
    SEMANTIC search for lot names using vector embeddings (pgvector / fallback FAISS).
    Use this when the user searches by a CATEGORY or CONCEPT (e.g. 'Канцелярия', 'Офисные товары',
    'Хозтовары') and ILIKE would miss related items (e.g. 'Степлер', 'Скрепки', 'Бумага').
    Returns the top semantically similar lot names that can then be searched for anomalies.
    """
    try:
        model = get_embedding_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0]

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Check if pgvector is available (vector column type)
        cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name='lot_embeddings' AND column_name='embedding';")
        col_type = cur.fetchone()

        if col_type and col_type[0] == 'USER-DEFINED':  # pgvector vector type
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
            cur.execute(
                "SELECT clean_name, 1 - (embedding <=> %s::vector) AS similarity FROM lot_embeddings ORDER BY embedding <=> %s::vector LIMIT %s;",
                (query_vec.tolist(), query_vec.tolist(), limit)
            )
        else:
            # Fallback: load all embeddings and compute cosine similarity in Python (FAISS-like)
            cur.execute("SELECT clean_name, embedding FROM lot_embeddings;")
            rows = cur.fetchall()
            if not rows:
                conn.close()
                return f"Таблица эмбеддингов пуста. Запустите build_embeddings.py."
            names = [r[0] for r in rows]
            # psycopg2 auto-parses JSONB to Python list — no json.loads() needed
            vecs = np.array([r[1] for r in rows], dtype=np.float32)
            sims = vecs @ query_vec  # cosine similarity (normalized vectors)
            top_k_idx = np.argsort(sims)[::-1][:limit]
            results = [(names[i], float(sims[i])) for i in top_k_idx]

            conn.close()
            lines = [f"Семантически похожие товары на '{query}' (по {limit} ближайших):"]
            for name, sim in results:
                lines.append(f"- '{name}' (сходство: {round(sim * 100, 1)}%)")
            lines.append("\n💡 Используй эти названия в других инструментах для поиска аномалий цены или объёма.")
            return "\n".join(lines)

        rows = cur.fetchall()
        conn.close()
        lines = [f"Семантически похожие товары на '{query}':"]
        for name, sim in rows:
            lines.append(f"- '{name}' (сходство: {round(sim * 100, 1)}%)")
        lines.append("\n💡 Используй эти названия в других инструментах для поиска аномалий цены или объёма.")
        return "\n".join(lines)

    except Exception as e:
        return f"Ошибка семантического поиска: {e}. Убедитесь, что `build_embeddings.py` был запущен."


@tool
def get_contract_info(query: str, limit: int = 5) -> str:
    """
    Search signed contracts (winning supplier, contract sum, sign date) by product name, supplier, or customer.
    Use this when the user asks WHO won a tender, what was the FINAL contract price (not the announced price),
    which SUPPLIER is most common, or who signed a specific contract.
    Works in Russian and Kazakh.
    """
    sql = """
        SELECT
            contract_number, sign_date::date, contract_sum,
            supplier_biin, supplier_name_ru,
            customer_name_ru,
            item_name, unit_price, quantity,
            trd_buy_id
        FROM contracts
        WHERE supplier_name_ru ILIKE %s
           OR customer_name_ru ILIKE %s
           OR item_name ILIKE %s
           OR trd_buy_name_ru ILIKE %s
        ORDER BY contract_sum DESC NULLS LAST
        LIMIT %s
    """
    term = f"%{query.strip()}%"
    data = execute_sql(sql, (term, term, term, term, limit))

    if not data or "error" in data[0]:
        return f"Договоры по запросу '{query}' не найдены в базе. Возможно, данные ещё не загружены (запустите load_contracts.py)."

    lines = [f"Найдено {len(data)} договор(ов) по запросу '{query}':"]
    for row in data:
        lines.append(
            f"- Договор №{row['contract_number']} от {row['sign_date']} | "
            f"Поставщик: {row['supplier_name_ru']} (БИИН: {row['supplier_biin']}) | "
            f"Заказчик: {row['customer_name_ru']} | "
            f"Сумма: {row['contract_sum']} тг | Товар: {row.get('item_name', '—')}\n"
            f"  [Объявление на портале](https://goszakup.gov.kz/ru/announce/index/{row['trd_buy_id']})"
        )
    return "\n".join(lines)


# --- 3. Agent Configuration ---

SYSTEM_PROMPT = '''
Ты — аналитический ИИ-агент «Бота Государственных закупок» для анализа данных портала государственных закупок Республики Казахстан.

🌐 ЯЗЫКИ: Ты понимаешь и отвечаешь на РУССКОМ и КАЗАХСКОМ языках.
- Если на русском (RU) — отвечай на русском.
- Если пользователь пишет на казахском (KZ) — отвечай на казахском.

Твоя задача — формировать понятные, читаемые и аргументированные ответы. 
ЗАПРЕЩЕНО: Использовать выделение жирным шрифтом (**), так как это мешает читаемости. Пиши обычным текстом.

ОБЩИЙ СТИЛЬ:
- Вместо технических терминов (например, "Лот №...") используй название Заказчика.
- Если завышение цены очень большое (более 1000%), пиши «в X раз» для наглядности (например, "в 10.5 раз").

РЕГИОНЫ И КАТО:
- Параметр region_name является НЕОБЯЗАТЕЛЬНЫМ. 
- Если пользователь НЕ указал регион, область или город — НЕ спрашивай его, а производи поиск по ВСЕЙ СТРАНЕ (оставляй region_name пустым).
- Если в вопросе упомянут регион (ВКО, Астана, Алматы и т.д.) — передавай его в region_name.

ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ОТВЕТА (ПИШИ ОБЫЧНЫМ ТЕКСТОМ, БЕЗ **):
1. Вердикт / Үкім: (Краткий вывод с ключевыми цифрами).
2. Использованные данные / Қолданылған деректер: (Период, фильтры, сущности, выборка).
3. Сравнение / Салыстыру: (Сравнение цен или объемов, медианы, отклонения. Упоминай заказчиков).
4. Метрика оценки / Бағалау метрикасы: (Пиши: "Метрика оценки: Справедливая цена (Fair Price)" или "Isolation Forest").
5. Ограничения и уверенность / Шектеулер мен сенімділік: (Степень надежности данных: Высокая/Средняя/Низкая).
6. Детализация / Егжей-тегжей: (Список подозрительных закупок с названиями организаций и ссылками).

СТРОГО ОТВЕЧАЙ НА ЯЗЫКЕ ЗАПРОСА. ЕСЛИ ВОПРОС НА КАЗАХСКОМ - ОТВЕЧАЙ ПОЛНОСТЬЮ НА КАЗАХСКОМ ЯЗЫКЕ!
'''


class AgentWrapper:
    """Wraps the new langchain 1.x create_agent graph to be compatible
    with the old AgentExecutor.invoke({"input": ..., "chat_history": ...}) interface."""

    def __init__(self, graph):
        self.graph = graph

    def invoke(self, inputs: dict) -> dict:
        from langchain_core.messages import HumanMessage, AIMessage
        user_input = inputs.get("input", "")
        history = inputs.get("chat_history", [])

        # Build messages list: history + new user message
        messages = list(history) + [HumanMessage(content=user_input)]

        result = self.graph.invoke({"messages": messages})

        # Extract last AI message as output
        all_messages = result.get("messages", [])
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage):
                return {"output": msg.content}
        return {"output": "Агент не вернул ответ."}


def get_agent_executor():
    """Returns an AgentWrapper compatible with the old .invoke() interface."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    tools = [
        get_anomalies_by_product,
        evaluate_org_fairness,
        evaluate_lot_fairness,
        get_volume_anomalies,
        get_ml_anomalies,
        semantic_lot_search,
        get_contract_info,
    ]
    graph = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return AgentWrapper(graph)

if __name__ == "__main__":
    # Test execution locally
    # os.environ["OPENAI_API_KEY"] = "your-key"
    try:
        executor = get_agent_executor()
        print("Agent initialized successfully.")
    except Exception as e:
        print("Setup error:", e)
