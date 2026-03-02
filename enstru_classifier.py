"""
Классификатор ENSTRU по тексту лота (через OpenAI LLM).
"""
import os
import json
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class EnstruPrediction(BaseModel):
    code: str = Field(description="Код ЕНСТРУ (например: 26.20.11.110)")
    probability: float = Field(description="Уверенность в предсказании от 0.0 до 1.0 (например: 0.95)")


class EnstruResponse(BaseModel):
    predictions: List[EnstruPrediction] = Field(description="Список предсказанных кодов")


def predict_enstru(
    text: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Возвращает top-k кодов ENSTRU и их вероятности (confidence) по тексту лота, используя LLM.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return []

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    parser = JsonOutputParser(pydantic_object=EnstruResponse)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты — эксперт по государственным закупкам Республики Казахстан. Твоя задача — классифицировать переданный текст лота (описание товара/услуги) и подобрать к нему наиболее подходящий код ЕНСТРУ (Единый номенклатурный справочник товаров, работ и услуг).\n\n"
                   "Верни ровно {k} наиболее подходящих кодов ЕНСТРУ. Код ЕНСТРУ обычно имеет формат XX.XX.XX.XXX или похожий.\n\n"
                   "Оцени свою уверенность (probability) от 0.0 до 1.0.\n\n"
                   "{format_instructions}"),
        ("user", "Текст лота: {text}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "text": clean_text,
            "k": k,
            "format_instructions": parser.get_format_instructions()
        })
        
        predictions = result.get("predictions", [])
        return [(p["code"], float(p["probability"])) for p in predictions]
        
    except Exception as e:
        print(f"Ошибка при классификации LLM: {e}")
        return []


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Классификатор ENSTRU по тексту лота (LLM)."
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="+",
        help="Текст лота (например, 'Ноутбук Acer, 16 ГБ ОЗУ').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Сколько топ-кодов ENSTRU вернуть.",
    )
    args = parser.parse_args()

    query = " ".join(args.text)
    preds = predict_enstru(query, k=args.k)
    if not preds:
        print("Не удалось классифицировать: пустой текст или ошибка LLM.")
        return

    print(f"Текст: {query}")
    print("Предсказанные ENSTRU коды (через LLM):")
    for code, prob in preds:
        print(f"  {code}  (p={prob:.3f})")


if __name__ == "__main__":
    cli()
