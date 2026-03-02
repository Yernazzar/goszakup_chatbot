import os
import logging
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "goszakup-secret-key-2024")

# Lazy init agent executor
_agent_executor = None

def get_executor():
    global _agent_executor
    if _agent_executor is None:
        try:
            from agent import get_agent_executor
            _agent_executor = get_agent_executor()
            logging.info("Agent executor initialized OK.")
        except Exception as e:
            logging.error(f"Failed to initialize agent: {e}")
            _agent_executor = None
    return _agent_executor


@app.route("/")
def index():
    # Reset chat history on page reload
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
            "reply": "❌ Агент недоступен. Проверьте OPENAI_API_KEY в файле .env и перезапустите сервер."
        })

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Format history for LangChain
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
        
        # Save to session history
        raw_history.append({"role": "user", "content": user_message})
        raw_history.append({"role": "assistant", "content": reply})
        session["history"] = raw_history[-10:] # Keep last 10 messages
        session.modified = True
        
    except Exception as e:
        logging.error(f"Agent error: {e}")
        reply = f"⚠️ Ошибка при обработке запроса: {e}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("=" * 60)
    print("  Бот Государственных закупок — Анализ госзакупок РК")
    print("  Веб-интерфейс: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
