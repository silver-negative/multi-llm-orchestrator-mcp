import asyncio
import os, logging, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, List, Dict

load_dotenv()

judge_llm = ChatOpenAI(
        model=os.environ["JUDGE_MODEL"],
        api_key=os.getenv("JUDGE_API_KEY", os.environ["API_KEY"]),
        base_url=os.getenv("JUDGE_BASE_URL", os.environ["BASE_URL"]),
        temperature=0,
        timeout=600,
        max_retries=0,
    )

JUDGE_PROMPT = """
    Вы — экспертный оценщик, сравнивающий два ответа ИИ.
    ## Критические инструкции:
    - Не отдавайте предпочтение ответам, основываясь на их длине.
    - Не отдавайте предпочтение ответам, основываясь на их позиции (первый или второй).
    - Сосредоточьтесь ИСКЛЮЧИТЕЛЬНО на качестве в соответствии с указанными критериями.
    - Ничьи допустимы, если ответы действительно эквивалентны.
    ## Инструкции:
    1. Сначала проанализируйте каждый ответ независимо.
    2. Сравните их по каждому критерию, используя следующую систему оценивания:
        - Оценка 5: фактически верно, прямо отвечает на вопрос, содержит конкретные детали. 
        - Оценка 4: в основном верно, но есть незначительная неточность.
        - Оценка 3: частично верно, но присутствуют ошибки, либо результат неполон
        - Оценка 2: в основном неверно, содержит крупные противоречия, 
        - Оценка 1: неверно или не по теме
    3. Определите общего победителя с указанием среднего уровня оценки каждого ответа.

    ## Критерии сравнения:
    - Фактическая точность (correctness)
    - Полнота (completeness)
    - Следование инструкциям (instruction-following)
    - Формат (format)

"""

class Judgement(BaseModel):
    winner: str = Field(description=""" Какой из ответов лучше: "A" | "B" | "Tie" """)
    confidence: float = Field(description= "Уверенность в сделанной оценке в пределах 0.0-1.0")
    A_score: float = Field(description= "Средняя оценка по критерия ответа A в пределах 1.0-5.0")
    B_score: float = Field(description= "Средняя оценка по критерия ответа B в пределах 1.0-5.0")

async def cast_judgement(task: str, response_a: str, response_b: str):
    user_prompt = f"""
        ## Исходный запрос:
        {task}
        
        ## Ответ A:
        {response_a}

        ## Ответ B:
        {response_b}   
    """
    verdict = judge_llm.with_structured_output(Judgement).ainvoke(([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=user_prompt)]))
    return verdict
 