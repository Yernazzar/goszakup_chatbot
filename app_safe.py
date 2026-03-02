#!/usr/bin/env python
"""
Безопасный запуск Flask с диагностикой ошибок.
Помогает найти и исправить проблемы со статус-кодами.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Загрузи .env ДО всех остальных импортов
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "=" * 70)
print("  ЗАПУСК Flask ДЛЯ ANTIGRAVITY")
print("=" * 70 + "\n")

# Проверка 1: Проверка окружения
print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ...")
print()

required_env = ['OPENAI_API_KEY']
missing = [key for key in required_env if not os.getenv(key)]

if missing:
    print(f"❌ Отсутствуют переменные окружения: {', '.join(missing)}")
    print(f"\n📝 Добавьте их в файл .env:")
    for key in missing:
        print(f"   {key}=value")
    sys.exit(1)

print("✅ Переменные окружения OK")
print()

# Проверка 2: Импорт зависимостей
print("🔍 ПРОВЕРКА ЗАВИСИМОСТЕЙ...")
print()

deps = {
    'flask': ('Flask', 'Flask'),
    'psycopg2': ('psycopg2', 'psycopg2'),
    'langchain_core': ('LangChain Core', 'langchain_core'),
    'langchain_openai': ('LangChain OpenAI', 'langchain_openai'),
}

for module_name, (display_name, import_name) in deps.items():
    try:
        __import__(import_name)
        print(f"  ✅ {display_name}")
    except ImportError as e:
        print(f"  ❌ {display_name}: {e}")
        print(f"\n     Установите: pip install {module_name}")
        sys.exit(1)

print()

# Проверка 3: Flask приложение
print("🔍 ИНИЦИАЛИЗАЦИЯ Flask...")
print()

try:
    from flask import Flask, render_template, request, jsonify, session
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "goszakup-secret-key-2024")
    print("  ✅ Flask приложение создано")
except Exception as e:
    print(f"  ❌ Ошибка инициализации Flask: {e}")
    sys.exit(1)

print()

# Проверка 4: Агент
print("🔍 ИНИЦИАЛИЗАЦИЯ АГЕНТА...")
print()

_agent_executor = None

def get_executor():
    global _agent_executor
    if _agent_executor is None:
        try:
            from agent import get_agent_executor
            _agent_executor = get_agent_executor()
            logger.info("✅ Агент инициализирован успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации агента: {e}")
            print(f"\n   ДЕТАЛЬ ОШИБКИ:")
            print(f"   {str(e)}")
            print(f"\n   РЕШЕНИЕ:")
            print(f"   1. Проверьте OPENAI_API_KEY в .env")
            print(f"   2. Проверьте подключение к БД (PostgreSQL запущен?)")
            print(f"   3. Запустите: python init_fair_price.py")
            _agent_executor = None
    return _agent_executor

# Попытка инициализировать агент при старте
executor = get_executor()
if executor is None:
    print("  ⚠️  Агент недоступен (попытка инициализации будет при первом запросе)")
else:
    print("  ✅ Агент готов")

print()

# Flask маршруты
@app.route("/")
def index():
    session["history"] = []
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Пустое сообщение"}), 400

    executor = get_executor()
    if executor is None:
        return jsonify({
            "reply": "❌ Агент недоступен. Проблемы:\n\n1. Проверьте OPENAI_API_KEY в .env\n2. Убедитесь PostgreSQL запущен\n3. Запустите: python init_fair_price.py\n\nПерезапустите сервер после исправлений."
        })

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        chat_history = []
        raw_history = session.get("history", [])
        for msg in raw_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        response = executor.invoke({
            "input": user_message,
            "chat_history": chat_history
        })
        reply = response.get("output", "Нет ответа от агента.")
        
        raw_history.append({"role": "user", "content": user_message})
        raw_history.append({"role": "assistant", "content": reply})
        session["history"] = raw_history[-10:]
        session.modified = True
        
    except Exception as e:
        logger.error(f"Agent error: {e}")
        reply = f"⚠️ Ошибка при обработке запроса: {e}"

    return jsonify({"reply": reply})

# Главная программа
if __name__ == "__main__":
    print("=" * 70)
    print("  ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ - СЕРВЕР ЗАПУСКАЕТСЯ")
    print("=" * 70)
    print()
    print("🌐 Откройте в браузере: http://127.0.0.1:5000")
    print()
    print("⚠️  ВНИМАНИЕ:")
    print("  - Закройте сервер: Ctrl+C")
    print("  - Логи ошибок будут выше")
    print()
    
    try:
        app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ЗАПУСКЕ: {e}")
        print("\nРешение:")
        print("  1. Проверьте порт 5000 не занят (netstat -an | grep 5000)")
        print("  2. Убедитесь все зависимости установлены (pip install -r requirements.txt)")
        print("  3. Запустите диагностику: python diagnose.py")
        sys.exit(1)
