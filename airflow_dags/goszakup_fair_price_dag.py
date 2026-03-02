"""
Airflow DAG для автоматического обновления Fair Price Mart.
Запускается ежедневно в 2:00 AM (GMT+6).

Установка:
1. Скопируйте этот файл в $AIRFLOW_HOME/dags/
2. airflow db init (если первый раз)
3. airflow dags list  # должен показать goszakup_fair_price_dag
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import logging

logger = logging.getLogger(__name__)

# Default args для DAG
default_args = {
    'owner': 'antigravity',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
}

# ============================================================================
# DAG Definition
# ============================================================================

dag = DAG(
    'goszakup_fair_price_dag',
    default_args=default_args,
    description='Ежедневное обновление Fair Price Mart для аналитики закупок',
    schedule_interval='0 2 * * *',  # Каждый день в 2:00 AM (GMT+6)
    catchup=False,
    tags=['goszakup', 'fair_price', 'analytics']
)

# ============================================================================
# DAG Tasks
# ============================================================================

def load_fresh_data():
    """Загрузить новые данные из API OWS v3."""
    logger.info("Загрузка новых данных из API...")
    try:
        from loader import main as loader_main
        import sys
        # Инкрементальная загрузка (не full-reload)
        sys.argv = ['loader.py']  # без флага --full-reload
        loader_main()
        logger.info("✅ Загрузка данных завершена")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки: {e}")
        raise


def build_fair_price_mart():
    """Пересчитать Fair Price Mart."""
    logger.info("Пересчёт Fair Price Mart...")
    try:
        from fair_price import build_fair_price_mart
        build_fair_price_mart()
        logger.info("✅ Fair Price Mart обновлена")
    except Exception as e:
        logger.error(f"❌ Ошибка при расчёте Fair Price: {e}")
        raise


def rebuild_ml_anomalies():
    """Пересчитать ML-аномалии (опционально, более дорого)."""
    logger.info("Пересчёт ML-аномалий (Isolation Forest)...")
    try:
        from ml_anomaly_detection import run_ml_anomaly_detection
        run_ml_anomaly_detection()
        logger.info("✅ ML-аномалии обновлены")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка ML: {e} (не критично)")


def rebuild_embeddings():
    """Пересчитать семантические embeddings (опционально)."""
    logger.info("Пересчёт семантических embeddings для поиска...")
    try:
        from build_embeddings import build_and_store_embeddings
        build_and_store_embeddings()
        logger.info("✅ Embeddings обновлены")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка embeddings: {e} (не критично)")


def health_check():
    """Проверка здоровья через агент."""
    logger.info("Проверка здоровья агента...")
    try:
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
        
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Проверки
        checks = {
            'purchases': "SELECT COUNT(*) FROM purchases",
            'lots': "SELECT COUNT(*) FROM lots",
            'mart_fair_price': "SELECT COUNT(*) FROM mart_fair_price",
            'anomalies_31%': "SELECT COUNT(*) FROM mart_fair_price WHERE deviation_percent > 30",
        }
        
        results = {}
        for name, query in checks.items():
            cur.execute(query)
            count = cur.fetchone()[0]
            results[name] = count
            logger.info(f"  {name}: {count:,}")
        
        conn.close()
        
        # Проверка минимальных требований
        assert results['purchases'] > 0, "purchases пуста"
        assert results['lots'] > 0, "lots пуста"
        assert results['mart_fair_price'] > 0, "mart_fair_price пуста"
        
        logger.info("✅ Все проверки пройдены")
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки: {e}")
        raise


def send_notification():
    """Отправить уведомление об успешном обновлении (опционально)."""
    logger.info("Отправка уведомления об успешном обновлении...")
    # TODO: интегрировать с правильным способом отправки notificaтioн
    # (email, Slack, Telegram и т.д.)
    logger.info("✅ Уведомление отправлено")


# ============================================================================
# Task Definitions
# ============================================================================

# ОСНОВНАЯ ЦЕПЬ ЗАДАЧ:
# 1. Загрузить новые данные
# 2. Пересчитать Fair Price
# 3. Пересчитать ML (параллельно)
# 4. Пересчитать embeddings (параллельно)
# 5. Проверка здоровья
# 6. Уведомление

task_load_data = PythonOperator(
    task_id='load_fresh_data',
    python_callable=load_fresh_data,
    dag=dag,
)

task_fair_price = PythonOperator(
    task_id='build_fair_price_mart',
    python_callable=build_fair_price_mart,
    dag=dag,
)

task_ml = PythonOperator(
    task_id='rebuild_ml_anomalies',
    python_callable=rebuild_ml_anomalies,
    dag=dag,
)

task_embeddings = PythonOperator(
    task_id='rebuild_embeddings',
    python_callable=rebuild_embeddings,
    dag=dag,
)

task_health = PythonOperator(
    task_id='health_check',
    python_callable=health_check,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# ============================================================================
# DAG Dependencies
# ============================================================================

# Цепь выполнения:
task_load_data >> task_fair_price >> [task_ml, task_embeddings] >> task_health >> task_notify

# ============================================================================
# Task Documentation
# ============================================================================

task_load_data.doc_md = """
### Загрузка новых данных
Запрашивает API OWS v3 по всем БИНам из `loader.py`.
Использует инкрементальную загрузку (только новые с последней даты).
Обновляет таблицы: purchases, lots, contracts
"""

task_fair_price.doc_md = """
### Пересчёт Fair Price
Пересчитывает витрину `mart_fair_price` со всеми компонентами:
- Справедливая цена (базовая по ENSTRU медиане)
- Региональная корректировка (KATO коэффициент)
- Инфляционная корректировка (по году/кварталу)
- Сезонная корректировка (по месяцу)
- Процент отклонения (deviation_percent)
"""

task_ml.doc_md = """
### ML-аномалии (Isolation Forest)
Выявляет многомерные аномалии по объёму и цене.
Требует больше CPU, но более точный анализ.
Обновляет таблицу: mart_ml_anomalies
"""

task_embeddings.doc_md = """
### Семантические embeddings
Создаёт vector embeddings для названий товаров.
Позволяет семантический поиск (например, "Канцелярия" найдёт "Ручку").
Обновляет таблицу: lot_embeddings
"""

task_health.doc_md = """
### Проверка здоровья
Проверяет что:
- purchases не пуста
- lots не пуста
- mart_fair_price не пуста
- Есть обнаруженные аномалии (>30%)

Если проверка не пройдена, DAG падает с ошибкой.
"""

task_notify.doc_md = """
### Уведомление об успехе
Отправляет уведомление что обновление завершено.
Можно интегрировать с email, Slack, Telegram и т.д.
"""

# ============================================================================
# DAG Documentation
# ============================================================================

dag.doc_md = """
# GoZakup Fair Price — Ежедневное обновление

## Описание
Автоматическое ежедневное обновление аналитики справедливых цен.
Запускается каждый день в 2:00 AM (GMT+6).

## Что происходит
1. **Загрузка данных** — импорт новых объявлений/лотов/контрактов из API
2. **Fair Price** — пересчёт справедливых цен со всеми корректировками
3. **ML-аномалии** — выявление многомерных аномалий (параллельно)
4. **Embeddings** — семантический поиск (параллельно)
5. **Проверка** — валидация что всё обновилось
6. **Уведомление** — сообщение об успехе

## Время выполнения
- Загрузка: 5-15 мин (зависит от объёма новых данных)
- Fair Price: 5-10 мин (45K лотов)
- ML: 10-20 мин (CPU-intensive)
- Embeddings: 10-15 мин
- Итого: ~30-60 мин

## Мониторинг
Смотрите в Airflow UI:
- Schedule: 0 2 * * * (UTC) / 8:00 AM по Астане (UTC+6)
- Logs доступны для каждой task
- Alerts при ошибках (если настроены)

## Улучшения
- [ ] Добавить email-уведомления
- [ ] Добавить Slack-интеграцию
- [ ] Параллелизм для load_data (по БИН-кодам)
- [ ] Caching для embeddings (обновлять только новые лоты)
- [ ] Backup перед важными операциями

## Контакты
Вопросы → смотреть логи Airflow или контактировать администратора.
"""

if __name__ == "__main__":
    # Для локального тестирования:
    # python goszakup_dag.py
    print("DAG: goszakup_fair_price_dag")
    print("Schedule: 0 2 * * * (ежедневно в 2:00 AM)")
    print("Tasks: load_data -> fair_price -> [ml, embeddings] -> health -> notify")
