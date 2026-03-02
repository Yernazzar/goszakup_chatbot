#!/usr/bin/env python
"""
Initialize Fair Price Mart and Reference Tables.
Run this ONCE after loading data to activate Fair Price calculations.

Usage:
    python init_fair_price.py
"""

import sys
import psycopg2
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

def main():
    print("=" * 70)
    print("  ИНИЦИАЛИЗАЦИЯ СПРАВЕДЛИВОЙ ЦЕНЫ (Fair Price)")
    print("=" * 70)
    print()
    
    try:
        # Verify connection
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        print("✅ Подключение к БД успешно")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        sys.exit(1)
    
    print()
    print("📋 ШАГИ ИНИЦИАЛИЗАЦИИ:")
    print()
    
    # Step 1: Import fair_price module
    print("1️⃣  Импортирование модуля fair_price...")
    try:
        from fair_price import initialize_reference_tables, build_fair_price_mart
        print("   ✅ Модуль загружен")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        sys.exit(1)
    
    # Step 2: Initialize reference tables
    print()
    print("2️⃣  Инициализация справочников (регионы, инфляция, сезонность)...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        initialize_reference_tables(conn)
        conn.close()
        print("   ✅ Справочники создано")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        sys.exit(1)
    
    # Step 3: Build Fair Price Mart
    print()
    print("3️⃣  Построение витрины данных (Fair Price Mart)...")
    print("   ⏳ Это может занять несколько минут...")
    try:
        build_fair_price_mart()
        print("   ✅ Витрина построена успешно")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        sys.exit(1)
    
    # Final verification
    print()
    print("4️⃣  Проверка результатов...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check mart_fair_price
        cur.execute("SELECT COUNT(*) FROM mart_fair_price;")
        lot_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM mart_fair_price WHERE deviation_percent > 30;")
        anomaly_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ref_regional_coefficients;")
        region_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ref_inflation_rates;")
        inflation_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ref_seasonality_factors;")
        season_count = cur.fetchone()[0]
        
        conn.close()
        
        print(f"   📊 Лотов обработано: {lot_count:,}")
        print(f"   🚨 Лотов с аномалиями (>30%): {anomaly_count:,}")
        print(f"   🌍 Регионов в справочнике: {region_count}")
        print(f"   📈 Периодов инфляции: {inflation_count}")
        print(f"   📅 Месяцев сезонности: {season_count}")
    except Exception as e:
        print(f"   ⚠️ Ошибка проверки: {e}")
    
    print()
    print("=" * 70)
    print("✅ ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print()
    print("📌 ВАЖНО:")
    print("  - Fair Price Mart готова к использованию")
    print("  - Агент теперь может анализировать справедливость цен")
    print("  - Все три фактора учитываются:")
    print("    1. Сравнительный подход (медиана по категории)")
    print("    2. Региональный коэффициент")
    print("    3. Инфляция и сезонность")
    print()
    print("▶️  Теперь запустите агент:")
    print("   python app.py")
    print()

if __name__ == "__main__":
    main()
