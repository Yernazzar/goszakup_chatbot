"""
Fair Price Calculation Module.
Implements Fair Price metric considering:
1. Comparative approach: Prices by similar ENSTRU codes
2. Regional coefficient: Adjustment based on delivery location (KATO code)
3. Temporal factor: Inflation and seasonality adjustments
"""

import psycopg2
import numpy as np
from sqlalchemy import create_engine, text
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv("DB_NAME", "goszakup_db"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD", "0000"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

DB_URI = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"


# ============================================================================
# 1. СПРАВОЧНИКИ И ИНИЦИАЛИЗАЦИЯ
# ============================================================================

def initialize_reference_tables(conn):
    """Create reference tables for regional coefficients and inflation rates."""
    cur = conn.cursor()
    
    # 1. REGIONAL COEFFICIENTS TABLE (KATO → region + coefficient)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_regional_coefficients (
            kato_code VARCHAR(10) PRIMARY KEY,
            region_name TEXT NOT NULL,
            region_code VARCHAR(5),
            coefficient NUMERIC(4,2) NOT NULL DEFAULT 1.0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Insert Kazakh regional coefficients based on official KATO classifier.
    # kato_code = first 2 digits of the 9-digit KATO code stored in purchases.kato_code.
    # E.g. kato_code '100000000' → prefix '10' → Абайская область
    # Coefficients are relative to г.Алматы (baseline = 1.0).
    regional_data = [
        ('10', 'Абайская область',             '10', 0.85, 'Восток / отдалённый регион'),
        ('11', 'Акмолинская область',           '11', 0.91, 'Центральный регион'),
        ('15', 'Актюбинская область',           '15', 0.92, 'Западный регион'),
        ('19', 'Алматинская область',           '19', 0.97, 'Пригород Алматы'),
        ('23', 'Атырауская область',            '23', 1.10, 'Нефтяной регион, высокие цены'),
        ('27', 'Западно-Казахстанская область', '27', 0.88, 'Западный регион'),
        ('31', 'Жамбылская область',            '31', 0.82, 'Южный регион'),
        ('33', 'область Жетісу',               '33', 0.90, 'Юго-восточный регион'),
        ('35', 'Карагандинская область',        '35', 0.95, 'Центральный регион'),
        ('39', 'Костанайская область',          '39', 0.91, 'Северный регион'),
        ('43', 'Кызылординская область',        '43', 0.87, 'Южный регион'),
        ('47', 'Мангистауская область',         '47', 1.15, 'Высокие логистические затраты'),
        ('51', 'Южно-Казахстанская область',    '51', 0.89, 'Южный регион'),
        ('55', 'Павлодарская область',          '55', 0.93, 'Северо-восточный регион'),
        ('59', 'Северо-Казахстанская область',  '59', 0.87, 'Северный регион'),
        ('61', 'Туркестанская область',         '61', 0.90, 'Южный регион'),
        ('62', 'область Ұлытау',               '62', 0.86, 'Центральный / новый регион'),
        ('63', 'Восточно-Казахстанская область','63', 0.94, 'Восточный регион'),
        ('71', 'г.Астана',                     '71', 1.20, 'Столица, высокие цены'),
        ('75', 'г.Алматы',                     '75', 1.00, 'Мегаполис — базовый уровень'),
        ('79', 'г.Шымкент',                    '79', 0.98, 'Крупный город'),
        ('00', 'Неизвестно',                   '00', 1.00, 'По умолчанию'),
    ]

    for kato, region, code, coef, note in regional_data:
        cur.execute("""
            INSERT INTO ref_regional_coefficients (kato_code, region_name, region_code, coefficient, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (kato_code) DO UPDATE
                SET region_name = EXCLUDED.region_name,
                    coefficient = EXCLUDED.coefficient,
                    notes       = EXCLUDED.notes
        """, (kato, region, code, coef, note))
    
    conn.commit()
    
    # 2. INFLATION RATES TABLE (year/quarter → inflation index)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_inflation_rates (
            year INT NOT NULL,
            quarter INT NOT NULL,
            inflation_index NUMERIC(4,3) NOT NULL,
            notes TEXT,
            PRIMARY KEY (year, quarter)
        );
    """)
    
    # Insert inflation indices (relative to 2023 Q1 = 1.0)
    # These are baseline indices for Kazakhstan
    inflation_data = [
        (2023, 1, 1.000, 'Baseline year'),
        (2023, 2, 1.025, 'Q2 2023'),
        (2023, 3, 1.045, 'Q3 2023'),
        (2023, 4, 1.055, 'Q4 2023'),
        (2024, 1, 1.070, 'Q1 2024 - Economic adjustment'),
        (2024, 2, 1.090, 'Q2 2024'),
        (2024, 3, 1.105, 'Q3 2024'),
        (2024, 4, 1.120, 'Q4 2024 - Year-end'),
        (2025, 1, 1.130, 'Q1 2025'),
        (2025, 2, 1.145, 'Q2 2025'),
        (2025, 3, 1.160, 'Q3 2025'),
        (2025, 4, 1.175, 'Q4 2025'),
        (2026, 1, 1.190, 'Q1 2026'),
    ]
    
    for year, quarter, index, note in inflation_data:
        cur.execute("""
            INSERT INTO ref_inflation_rates (year, quarter, inflation_index, notes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (year, quarter) DO UPDATE SET inflation_index = %s
        """, (year, quarter, index, note, index))
    
    conn.commit()
    
    # 3. SEASONALITY FACTORS TABLE (month → seasonality factor)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_seasonality_factors (
            month INT PRIMARY KEY,
            seasonality_factor NUMERIC(4,3) NOT NULL,
            notes TEXT
        );
    """)
    
    # Insert seasonality factors (relative to annual average = 1.0)
    seasonality_data = [
        (1, 1.05, 'January - slightly elevated demand'),
        (2, 0.98, 'February - lower demand'),
        (3, 1.02, 'March - spring restocking'),
        (4, 1.08, 'April - procurement season begins'),
        (5, 1.12, 'May - high procurement activity'),
        (6, 1.10, 'June - end of Q1 budget cycle'),
        (7, 0.95, 'July - summer slow period'),
        (8, 0.92, 'August - vacation period'),
        (9, 1.05, 'September - new fiscal year start'),
        (10, 1.15, 'October - autumn procurement peak'),
        (11, 1.18, 'November - year-end budget clearing'),
        (12, 1.20, 'December - fiscal year-end rush'),
    ]
    
    for month, factor, note in seasonality_data:
        cur.execute("""
            INSERT INTO ref_seasonality_factors (month, seasonality_factor, notes)
            VALUES (%s, %s, %s)
            ON CONFLICT (month) DO UPDATE SET seasonality_factor = %s
        """, (month, factor, note, factor))
    
    conn.commit()
    print("✅ Reference tables initialized successfully!")


# ============================================================================
# 2. FAIR PRICE CALCULATION FUNCTIONS
# ============================================================================

def get_inflation_index(year: int, month: int) -> float:
    """Get inflation adjustment index for given year/month."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    quarter = (month - 1) // 3 + 1
    
    cur.execute("""
        SELECT inflation_index 
        FROM ref_inflation_rates 
        WHERE year = %s AND quarter = %s
    """, (year, quarter))
    
    result = cur.fetchone()
    conn.close()
    
    if result:
        return float(result[0])
    else:
        # Fallback to most recent known index
        return 1.0


def get_regional_coefficient(kato_code: str) -> float:
    """Get regional price adjustment coefficient based on KATO code.
    
    Accepts both the full 9-digit KATO code stored in purchases.kato_code
    (e.g. '100000000') and the 2-digit prefix (e.g. '10').
    Matching is done by comparing the first 2 characters of kato_code
    against ref_regional_coefficients.kato_code.
    """
    if not kato_code:
        kato_code = '00'

    # Normalise: take first 2 digits of whatever length is passed
    prefix = str(kato_code).strip()[:2].zfill(2)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT coefficient
        FROM ref_regional_coefficients
        WHERE kato_code = %s
    """, (prefix,))

    result = cur.fetchone()
    conn.close()

    if result:
        return float(result[0])
    else:
        return 1.0


def get_seasonality_factor(month: int) -> float:
    """Get seasonality adjustment factor for given month."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT seasonality_factor 
        FROM ref_seasonality_factors 
        WHERE month = %s
    """, (month,))
    
    result = cur.fetchone()
    conn.close()
    
    if result:
        return float(result[0])
    else:
        return 1.0


def calculate_baseline_fair_price(enstru_code: str, year: int, kato_code: str = None, clean_name: str = None) -> Dict[str, float]:
    """
    Calculate baseline fair price (median) for a product category (ENSTRU)
    for a given year and region.
    
    Returns:
        dict with 'median_price', 'count', 'q25', 'q75'
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Filter by ENSTRU code or clean_name if ENSTRU is UNKNOWN
    if enstru_code and enstru_code != 'UNKNOWN':
        where_clause = "WHERE c.enstru_code = %s AND EXTRACT(YEAR FROM p.publish_date)::INT = %s"
        params = [enstru_code, year]
    elif clean_name:
        where_clause = "WHERE c.clean_name = %s AND EXTRACT(YEAR FROM p.publish_date)::INT = %s"
        params = [clean_name, year]
    else:
        # Fallback to search NULL enstru_code
        where_clause = "WHERE c.enstru_code IS NULL AND EXTRACT(YEAR FROM p.publish_date)::INT = %s"
        params = [year]

    if kato_code:
        where_clause += " AND p.kato_code = %s"
        params.append(kato_code)
    
    # Get median and quartiles of unit_price for this category
    cur.execute(f"""
        SELECT 
            COUNT(*) as count,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY c.unit_price) as q25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY c.unit_price) as q50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY c.unit_price) as q75,
            AVG(c.unit_price) as avg_price
        FROM core_lots_cleaned c
        JOIN purchases p ON c.purchase_id = p.id
        {where_clause}
        AND c.unit_price > 0
    """, params)
    
    result = cur.fetchone()
    conn.close()
    
    if result and result[0] > 0:
        return {
            'count': result[0],
            'q25': float(result[1]) if result[1] else None,
            'median_price': float(result[2]) if result[2] else None,
            'q75': float(result[3]) if result[3] else None,
            'avg_price': float(result[4]) if result[4] else None
        }
    else:
        return {'count': 0, 'median_price': None}


def calculate_fair_price(unit_price: float, enstru_code: str, year: int, month: int, 
                         kato_code: str = None, clean_name: str = None) -> Dict[str, Any]:
    """
    Calculate Fair Price with all adjustments:
    Fair Price = Baseline Price × Regional Coefficient × Inflation Index × Seasonality Factor
    
    Args:
        unit_price: actual lot unit price
        enstru_code: product category code
        year: purchase year
        month: purchase month
        kato_code: regional KATO code
        clean_name: fallback name if enstru_code is missing
    
    Returns:
        dict with 'fair_price', 'baseline_price', 'regional_coef', 'inflation_index', 
              'seasonality_factor', 'deviation_percent'
    """
    
    # 1. Get baseline fair price (median for this category/year/region)
    baseline_data = calculate_baseline_fair_price(enstru_code, year, kato_code, clean_name)
    baseline_price = baseline_data.get('median_price')
    
    if not baseline_price or baseline_price <= 0:
        # Fallback: use avg_price if median not available
        baseline_price = baseline_data.get('avg_price')
        if not baseline_price:
            return {
                'fair_price': unit_price,
                'baseline_price': None,
                'regional_coef': 1.0,
                'inflation_index': 1.0,
                'seasonality_factor': 1.0,
                'deviation_percent': 0.0,
                'sample_count': 0,
                'confidence': 'Низкая - недостаточно данных для сравнения'
            }
    
    # 2. Get regional coefficient
    regional_coef = get_regional_coefficient(kato_code)
    
    # 3. Get inflation adjustment
    inflation_index = get_inflation_index(year, month)
    
    # 4. Get seasonality factor
    seasonality_factor = get_seasonality_factor(month)
    
    # 5. Calculate Fair Price
    fair_price = baseline_price * regional_coef * inflation_index * seasonality_factor
    
    # 6. Calculate deviation
    if fair_price > 0:
        deviation_percent = ((unit_price - fair_price) / fair_price) * 100
    else:
        deviation_percent = 0.0
    
    # Determine confidence level based on sample size
    sample_count = baseline_data.get('count', 0)
    if sample_count >= 50:
        confidence = 'Высокая'
    elif sample_count >= 20:
        confidence = 'Средняя'
    else:
        confidence = 'Низкая'
    
    return {
        'fair_price': round(fair_price, 2),
        'baseline_price': round(baseline_price, 2),
        'regional_coef': round(regional_coef, 3),
        'inflation_index': round(inflation_index, 4),
        'seasonality_factor': round(seasonality_factor, 3),
        'deviation_percent': round(deviation_percent, 2),
        'sample_count': sample_count,
        'confidence': confidence
    }


# ============================================================================
# 3. MART GENERATION (Create analytical marts with Fair Price)
# ============================================================================

def build_fair_price_mart():
    """Build mart_fair_price table with Fair Price calculations for all lots."""
    print("Building Fair Price Mart...")
    
    engine = create_engine(DB_URI)
    conn = engine.connect()
    
    # Drop existing mart
    conn.execute(text("DROP TABLE IF EXISTS mart_fair_price;"))
    
    # Create new mart with Fair Price calculations
    conn.execute(text("""
        CREATE TABLE mart_fair_price AS
        SELECT 
            c.id as lot_id,
            p.id as purchase_id,
            p.number_anno,
            p.org_bin,
            p.org_name_ru,
            c.clean_name,
            COALESCE(c.enstru_code, 'UNKNOWN') as enstru_code,
            p.kato_code,
            r.region_name,
            c.unit_price,
            c.count as quantity,
            c.amount as total_amount,
            EXTRACT(YEAR FROM p.publish_date)::INT as purchase_year,
            EXTRACT(MONTH FROM p.publish_date)::INT as purchase_month,
            p.publish_date,
            -- Fair Price components will be calculated via Python
            0::NUMERIC as fair_price,
            0::NUMERIC as baseline_price,
            1.0::NUMERIC as regional_coef,
            1.0::NUMERIC as inflation_index,
            1.0::NUMERIC as seasonality_factor,
            0.0::NUMERIC as deviation_percent,
            0 as sample_count
        FROM core_lots_cleaned c
        JOIN purchases p ON c.purchase_id = p.id
        -- Match by first 2 digits of the 9-digit KATO code (official KZ classifier)
        LEFT JOIN ref_regional_coefficients r
               ON LEFT(p.kato_code::TEXT, 2) = r.kato_code
        WHERE c.unit_price > 0 AND c.count > 0;
    """))
    
    conn.commit()
    
    # Now calculate Fair Price for each lot
    print("Calculating Fair Price for each lot...")
    
    # Fetch all lots
    result = conn.execute(text("""
        SELECT lot_id, clean_name, enstru_code, purchase_year, purchase_month, kato_code, unit_price
        FROM mart_fair_price
        ORDER BY purchase_year DESC, lot_id
    """))
    
    lots = result.fetchall()
    total = len(lots)
    
    # Batch update
    from typing import Dict
    updates = []
    
    for idx, (lot_id, clean_name, enstru_code, year, month, kato_code, unit_price) in enumerate(lots):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{total}...")
        
        fair_calc = calculate_fair_price(
            unit_price=unit_price,
            enstru_code=enstru_code,
            year=year,
            month=month,
            kato_code=kato_code,
            clean_name=clean_name
        )
        
        conn.execute(text("""
            UPDATE mart_fair_price 
            SET 
                fair_price = :fair_price,
                baseline_price = :baseline_price,
                regional_coef = :regional_coef,
                inflation_index = :inflation_index,
                seasonality_factor = :seasonality_factor,
                deviation_percent = :deviation_percent,
                sample_count = :sample_count
            WHERE lot_id = :lot_id
        """), {
            'fair_price': fair_calc['fair_price'],
            'baseline_price': fair_calc['baseline_price'],
            'regional_coef': fair_calc['regional_coef'],
            'inflation_index': fair_calc['inflation_index'],
            'seasonality_factor': fair_calc['seasonality_factor'],
            'deviation_percent': fair_calc['deviation_percent'],
            'sample_count': fair_calc['sample_count'],
            'lot_id': lot_id
        })
    
    conn.commit()
    
    # Create indices for fast querying
    conn.execute(text("CREATE INDEX idx_fair_price_enstru ON mart_fair_price(enstru_code);"))
    conn.execute(text("CREATE INDEX idx_fair_price_org ON mart_fair_price(org_bin);"))
    conn.execute(text("CREATE INDEX idx_fair_price_deviation ON mart_fair_price(deviation_percent);"))
    conn.execute(text("CREATE INDEX idx_fair_price_region ON mart_fair_price(kato_code);"))
    
    conn.commit()
    conn.close()
    
    print("✅ Fair Price Mart built successfully!")


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    initialize_reference_tables(conn)
    conn.close()
    
    build_fair_price_mart()
    
    print("\n✅ Fair Price Pipeline completed!")
    print("\nNow you can use mart_fair_price for:")
    print("- Finding price anomalies with regional/temporal adjustments")
    print("- Comparing fairness of purchases across regions")
    print("- Analyzing seasonality effects on procurement")
    print("- Detecting systematic overpricings with confidence levels")
