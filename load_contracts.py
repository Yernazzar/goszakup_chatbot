"""
Contract API loader — fetches signed contracts for all organisations from OWS v3.
Field names verified via GraphQL introspection on 2026-02-26.
Can be run standalone: python load_contracts.py
Or called from loader.py main().
"""
import requests
import psycopg2
import time
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "goszakup_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "0000"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

TOKEN = os.getenv("GOSZAKUP_API_TOKEN")
URL = "https://ows.goszakup.gov.kz/v3/graphql"

BINS = [
    '000740001307', '020240002363', '020440003656', '030440003698',
    '050740004819', '051040005150', '100140011059', '120940001946', '140340016539',
    '150540000186', '171041003124', '210240019348', '210240033968', '210941010761',
    '230740013340', '231040023028', '780140000023', '900640000128', '940740000911',
    '940940000384', '960440000220', '970940001378', '971040001050', '980440001034',
    '981140001551', '990340005977', '990740002243'
]

# Field names verified via introspection — all camelCase as returned by the API
CONTRACTS_QUERY = """
query getContracts($bin: String, $after: Int) {
  Contract(filter: {customerBin: $bin}, limit: 50, after: $after) {
    id
    contractNumber
    signDate
    contractSum
    contractSumWnds
    supplierBiin
    supplierFio
    customerBin
    trdBuyId
    trdBuyNameRu
    finYear
    refContractStatusId
    Supplier {
      nameRu
    }
    Customer {
      nameRu
    }
    ContractUnits {
      itemPrice
      quantity
      totalSum
    }
  }
}
"""


def fetch_contracts_page(bin_code: str, after=None, retries=5):
    variables = {"bin": bin_code, "after": after}
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            response = requests.post(
                URL,
                json={"query": CONTRACTS_QUERY, "variables": variables},
                headers=headers,
                timeout=40,
            )
            response.raise_for_status()
            result = response.json()

            # Log any GraphQL errors to help debug
            if "errors" in result:
                for err in result["errors"]:
                    print(f"  [GraphQL Error] {err.get('message')}")
                return None  # stop on schema errors

            return result

        except Exception as e:
            if attempt < retries - 1:
                wait = (attempt + 1) * 3
                print(f"  [!] Retry {attempt+1}/{retries} for BIN {bin_code}: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [X] Failed after {retries} retries for BIN {bin_code}")
                return None


def load_contracts():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Create table if it doesn't exist yet (new deployments)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id BIGINT PRIMARY KEY,
            contract_number TEXT,
            sign_date TIMESTAMP,
            contract_sum NUMERIC,
            supplier_biin TEXT,
            supplier_fio TEXT,
            supplier_name_ru TEXT,
            customer_bin TEXT,
            customer_name_ru TEXT,
            trd_buy_id BIGINT,
            trd_buy_name_ru TEXT,
            contract_year INT,
            status_id INT,
            item_name TEXT,
            unit_price NUMERIC,
            quantity NUMERIC
        );
    """)

    # Migrate old column names if they still exist (idempotent)
    cur.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='contracts' AND column_name='number_contract'
            ) THEN
                ALTER TABLE contracts RENAME COLUMN number_contract TO contract_number;
            END IF;
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='contracts' AND column_name='supplier_bin'
            ) THEN
                ALTER TABLE contracts RENAME COLUMN supplier_bin TO supplier_biin;
            END IF;
        END $$;
    """)
    # Add missing columns (idempotent)
    cur.execute("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS supplier_fio TEXT;")
    cur.execute("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS trd_buy_name_ru TEXT;")
    cur.execute("ALTER TABLE contracts ADD COLUMN IF NOT EXISTS lot_id BIGINT;")  # keep for compatibility


    conn.commit()

    total_contracts = 0

    for bin_code in BINS:
        print(f"\n  Контракты БИН: {bin_code}", end="", flush=True)
        last_id = None
        has_next = True
        bin_count = 0

        while has_next:
            result = fetch_contracts_page(bin_code, last_id)
            if not result or "data" not in result or result["data"] is None:
                break

            contracts = result["data"].get("Contract", [])
            page_info = result.get("extensions", {}).get("pageInfo", {})

            if not contracts:
                break

            for c in contracts:
                # Supplier info: prefer relation nameRu, fallback to supplierFio
                supplier_name = None
                if c.get("Supplier") and c["Supplier"].get("nameRu"):
                    supplier_name = c["Supplier"]["nameRu"]
                elif c.get("supplierFio"):
                    supplier_name = c["supplierFio"]

                # Customer name from relation
                customer_name = None
                if c.get("Customer") and c["Customer"].get("nameRu"):
                    customer_name = c["Customer"]["nameRu"]

                # First ContractUnit for price/qty details
                units = c.get("ContractUnits") or []
                item_name = c.get("trdBuyNameRu")            # ContractUnits has no nameRu
                unit_price = float(units[0].get("itemPrice") or 0) if units else 0
                quantity = float(units[0].get("quantity") or 0) if units else 0

                cur.execute("""
                    INSERT INTO contracts (
                        id, contract_number, sign_date, contract_sum,
                        supplier_biin, supplier_fio, supplier_name_ru,
                        customer_bin, customer_name_ru,
                        trd_buy_id, trd_buy_name_ru,
                        contract_year, status_id,
                        item_name, unit_price, quantity
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE
                        SET contract_sum = EXCLUDED.contract_sum,
                            status_id = EXCLUDED.status_id,
                            supplier_name_ru = EXCLUDED.supplier_name_ru
                """, (
                    c["id"],
                    c.get("contractNumber"),
                    c.get("signDate"),
                    c.get("contractSum"),
                    c.get("supplierBiin"),
                    c.get("supplierFio"),
                    supplier_name,
                    c.get("customerBin"),
                    customer_name,
                    c.get("trdBuyId"),
                    c.get("trdBuyNameRu"),
                    c.get("finYear"),
                    c.get("refContractStatusId"),
                    item_name,
                    unit_price,
                    quantity
                ))

            conn.commit()
            bin_count += len(contracts)
            total_contracts += len(contracts)

            has_next = page_info.get("hasNextPage", False)
            last_id = page_info.get("lastId")
            time.sleep(0.5)

        print(f" — {bin_count} договоров")

    conn.close()
    print(f"\n✅ Загружено договоров: {total_contracts}")


if __name__ == "__main__":
    load_contracts()
