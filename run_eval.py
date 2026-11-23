import asyncio
import os, logging, json, shutil
from datetime import datetime
from pathlib import Path
from typing import get_args
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import get_usage_metadata_callback
from orchestrator import TaskState, create_app
from orchestrator.config import (get_planner_llm, get_executor_llm,
                                 get_subtask_reviewer_llm, get_reviewer_llm)
from eval.judge import cast_judgement, judge_llm, CategoryName
try:
    from eval.pricing import load_pricing, compute_costs
except ImportError:
    # eval/pricing.py отсутствует — стоимость не считаем (поля cost_rub будут пустыми)
    def load_pricing(path=None):
        return {}

    def compute_costs(model_usage, pricing):
        for u in model_usage.values():
            u["cost_rub"] = None
        return 0.0, list(model_usage)

load_dotenv()

# Количество прогонов каждой задачи (для оценки дисперсии результатов).
# Пока 1 — лишние токены не тратим. Управляется переменной окружения EVAL_RUNS.
RUNS = int(os.getenv("EVAL_RUNS", "1"))

# Пути. Базовая директория — рядом с этим скриптом; EVAL_DIR можно переопределить
# (используется в тестах, чтобы не писать в рабочий eval/).
EVAL_DIR = Path(os.getenv("EVAL_DIR", Path(__file__).resolve().parent / "eval"))
TASKS_FILE = EVAL_DIR / "tasks.jsonl"
RUNS_DIR = EVAL_DIR / "runs"

# Нормализация вердикта судьи. Судья видит только «Ответ A» / «Ответ B» и НЕ ЗНАЕТ,
# какому решению принадлежит ответ: в прямом вызове A — пайплайн, в обратном — singular.
WINNER_BY_FWD = {"A": "pipeline", "B": "singular", "Tie": "tie"}
WINNER_BY_BWD = {"A": "singular", "B": "pipeline", "Tie": "tie"}

def scores_by_name(score_list):
    """Список оценок судьи -> {имя категории: балл}. Порядок ответа модели не важен."""
    return {s.category: s.score for s in score_list}

def average_scores(*dicts):
    """Усреднение оценок по ИМЕНАМ категорий: одинаковые категории складываются,
    даже если судья вернул их в разном порядке. Пропущенные категории не учитываются."""
    result = {}
    for cat in get_args(CategoryName):
        values = [d[cat] for d in dicts if d and cat in d]
        if values:
            result[cat] = sum(values) / len(values)
    return result

def normalize_winner(verdict_winner, reverse_winner):
    """(fwd, bwd, final) в метках pipeline/singular/tie по обоим вердиктам.
    Расхождение прямого и обратного прохода трактуем как ничью (позиционное смещение судьи)."""
    fwd = WINNER_BY_FWD[verdict_winner]
    bwd = WINNER_BY_BWD[reverse_winner]
    return fwd, bwd, (fwd if fwd == bwd else "tie")

def decide_without_judge(final_answer_A: str, final_answer_B: str):
    """Пара с пустым ответом обходится без судьи: победитель по правилу, оценок нет."""
    a_empty = not final_answer_A.strip()
    b_empty = not final_answer_B.strip()
    if a_empty and b_empty:
        return "tie", "оба ответа пусты"
    if a_empty:
        return "singular", "ответ пайплайна пуст"
    return "pipeline", "ответ singular пуст"

async def judge_with_retry(task: str, response_a: str, response_b: str, max_attempts: int = 2):
    """Вызов судьи с повторами: временный сбой судьи не должен ронять прогон.

    Повторяет неудавшийся вызов (как ainvoke_with_retry у LLM); если судья недоступен
    совсем — исключение уходит наверх, где гасится в заглушку без оценки.
    """
    import orchestrator.routing as rt_mod
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await cast_judgement(task, response_a, response_b)
        except Exception as e:
            last_error = e
            print(f"[judge] попытка {attempt}/{max_attempts} не удалась: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(rt_mod.RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise RuntimeError(f"судья не ответил после {max_attempts} попыток: {last_error}") from last_error

def model_params(llm, fallback_model=None, temperature=None):
    """Параметры модели для params.json (без каких-либо ключей доступа)."""
    return {
        "model": getattr(llm, "model_name", None) or fallback_model,
        "temperature": getattr(llm, "temperature", temperature),
    }

def make_run_dir():
    """Новая папка прогона eval/runs/<timestamp>."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir

async def main():
    orchestrator_app = create_app()

    singular_llm = ChatOpenAI(
        model=os.environ["PLANNER_MODEL"],
        api_key=os.getenv("PLANNER_API_KEY", os.environ["API_KEY"]),
        base_url=os.getenv("PLANNER_BASE_URL", os.environ["BASE_URL"]),
        temperature=0.3,
        timeout=600,
        max_retries=0,
    )

    # --- Папка прогона: результаты, копия задач и параметры моделей ---
    run_id, run_dir = make_run_dir()
    shutil.copy(TASKS_FILE, run_dir / "tasks.jsonl")
    params = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs_per_task": RUNS,
        "models": {
            "pipeline": {
                "planner":          model_params(get_planner_llm(),          os.getenv("PLANNER_MODEL")),
                "executor":         model_params(get_executor_llm(),         os.getenv("EXECUTOR_MODEL")),
                "subtask_reviewer": model_params(get_subtask_reviewer_llm(),  os.getenv("SUB_REVIEWER_MODEL")),
                "reviewer":         model_params(get_reviewer_llm(),         os.getenv("REVIEWER_MODEL")),
            },
            "singular": model_params(singular_llm, os.getenv("PLANNER_MODEL"), temperature=0.3),
            "judge":    model_params(judge_llm, os.getenv("JUDGE_MODEL"), temperature=0),
        },
    }
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    pricing = load_pricing()
    print(f"Прогон {run_id}: результаты сохраняются в {run_dir}")

    with    (open(TASKS_FILE, 'r', encoding='utf-8') as f, 
            open(run_dir / 'pipeline_results.jsonl', 'w+', encoding='utf-8') as p,
            open(run_dir / 'singular_results.jsonl', 'w+', encoding='utf-8') as s,
            open(run_dir / 'judge_summary.jsonl', 'w+', encoding='utf-8') as j):

        for line in f:
            if not line.strip():
                continue
            line_dict = json.loads(line)
            task = line_dict["task"]

            for run in range(1, RUNS + 1):
                print(f'Задание №{line_dict["id"]} (прогон {run}/{RUNS}): {task}')

                initial_state: TaskState = {
                    "original_task": task,
                    "plan": [],
                    "depends_on": [],
                    "results": [],
                    "subtask_reviews": [],
                    "subtask_statuses": [],
                    "feedback": None,
                    "final_answer": None,
                    "revision_count": 0,
                }
                usage_cb = None

                # ---------- A: пайплайн ----------
                final_answer_A = ""
                task_failed = False
                technical_fail = False
                revision_count = 0
                model_usage = {}
                try:
                    with get_usage_metadata_callback() as usage_cb:
                        result = await orchestrator_app.ainvoke(initial_state)
                    
                    if result.get("final_answer"):
                        final_answer_A = result["final_answer"]
                        revision_count = result['revision_count']
                except Exception as e:
                    technical_fail = True
                    print(f"[pipeline] ошибка: {e}")
                finally:
                    # usage читаем и при успехе, и при падении: частично потраченные токены не теряем
                    if usage_cb is not None:
                        for model, usage in usage_cb.usage_metadata.items():
                            inp = usage.get("input_tokens", 0)
                            out = usage.get("output_tokens", 0)
                            cache = (usage.get("input_token_details") or {}).get("cache_read", 0)
                            model_usage[model] = {"input": inp, "output": out, "cache_read": cache}
                if final_answer_A.strip() == "":
                    task_failed = True

                cost_rub, unknown = compute_costs(model_usage, pricing)
                for m in unknown:
                    print(f"[pricing] нет цены для модели {m} — дополните eval/pricing.json")

                pipeline_summary = { 
                    "id": line_dict["id"],
                    "run": run,
                    "no_response": technical_fail,
                    "failed": task_failed,
                    "rev_count": revision_count,
                    "model_usage": model_usage,
                    "cost_rub": cost_rub,
                    "result": final_answer_A
                }

                # ---------- B: одна LLM ----------

                usage_cb = None
                final_answer_B = ""
                task_failed = False
                technical_fail = False
                model_usage = {}
                try:
                    with get_usage_metadata_callback() as usage_cb:
                        result = await singular_llm.ainvoke(task)
                    
                    if result:
                        final_answer_B = result.content
                except Exception as e:
                    technical_fail = True
                    print(f"[singular] ошибка: {e}")
                finally:
                    if usage_cb is not None:
                        for model, usage in usage_cb.usage_metadata.items():
                            inp = usage.get("input_tokens", 0)
                            out = usage.get("output_tokens", 0)
                            cache = (usage.get("input_token_details") or {}).get("cache_read", 0)
                            model_usage[model] = {"input": inp, "output": out, "cache_read": cache}
                if final_answer_B.strip() == "":
                    task_failed = True

                cost_rub, unknown = compute_costs(model_usage, pricing)
                for m in unknown:
                    print(f"[pricing] нет цены для модели {m} — дополните eval/pricing.json")
                
                singular_summary = { 
                    "id": line_dict["id"],
                    "run": run,
                    "no_response": technical_fail,
                    "failed": task_failed,
                    "model_usage": model_usage,
                    "cost_rub": cost_rub,
                    "result": final_answer_B
                }

                # ---------- Судья ----------
                # Пустой ответ в паре: судью не вызываем, победитель по правилу, оценок нет.
                #
                # Текущее решение (бюджет сэкономлен): scores = null у обоих ответов. Пара
                # исключается из средних по категориям, но остаётся в win-rate (пустой ответ —
                # проигрыш, оба пустые — ничья). При агрегации обязательно показывать покрытие:
                # «оценено N из M задач» — иначе средние систематически завышаются.
                #
                # ОПТИМАЛЬНОЕ решение, когда позволяет бюджет:
                #   1) пустому ответу сразу выставлять scores = 1 по всем категориям (штраф:
                #      худший возможный исход на шкале 1–5, судья бы и не поставил больше);
                #   2) непустой ответ оценивать ОДНОСТОРОННИМ судейством — один вызов судьи
                #      с абсолютной рубрикой 1–5 по категориям, без пары: это и дешевле
                #      парного сравнения (1 вызов вместо 2), и избавлено от позиционного
                #      смещения (сравнивать не с чем);
                #   3) для задач с verifiable/expected в tasks.jsonl судью не звать вообще —
                #      заменить детерминированной проверкой (regex «Ответ: N», прогон кода):
                #      дешевле и объективнее любого судьи.
                if not final_answer_A.strip() or not final_answer_B.strip():
                    final_winner, rule_note = decide_without_judge(final_answer_A, final_answer_B)
                    judge_summary = {
                        "id": line_dict["id"],
                        "run": run,
                        "judge_called": False,
                        "winner": {"fwd": None, "bwd": None, "final": final_winner},
                        "confidence": {"fwd": None, "bwd": None},
                        "rationale": {"fwd": None, "bwd": None,
                                      "rule": f"судья не вызывался: {rule_note}"},
                    }
                    pipeline_summary["scores"] = None
                    singular_summary["scores"] = None
                else:
                    try:
                        verdict =           await judge_with_retry(task, final_answer_A, final_answer_B)
                        reverse_verdict =   await judge_with_retry(task, final_answer_B, final_answer_A)
                    except Exception as e:
                        # Судья недоступен и после ретраев: прогон не роняем, пара остаётся без оценки
                        print(f"[judge] судья недоступен, пара не оценена: {e}")
                        judge_summary = {
                            "id": line_dict["id"],
                            "run": run,
                            "judge_called": False,
                            "winner": {"fwd": None, "bwd": None, "final": "tie"},
                            "confidence": {"fwd": None, "bwd": None},
                            "rationale": {"fwd": None, "bwd": None,
                                          "rule": f"судья не ответил после ретраев: {e}"},
                        }
                        pipeline_summary["scores"] = None
                        singular_summary["scores"] = None
                    else:
                        fwd_winner, bwd_winner, final_winner = normalize_winner(
                            verdict.winner, reverse_verdict.winner)

                        judge_summary = {
                            "id": line_dict["id"],
                            "run": run,
                            "judge_called": True,
                            "winner":   {"fwd": fwd_winner,
                                         "bwd": bwd_winner,
                                         "final": final_winner},
                            "confidence":   {"fwd": verdict.confidence,
                                             "bwd": reverse_verdict.confidence},
                            "rationale":    {"fwd": verdict.rationale,
                                             "bwd": reverse_verdict.rationale}
                        }

                        # Усреднение по именам категорий: в прямом проходе пайплайн — это A,
                        # в обратном — B (и наоборот для singular)
                        pipeline_summary["scores"] = average_scores(
                            scores_by_name(verdict.A_scores),
                            scores_by_name(reverse_verdict.B_scores))
                        singular_summary["scores"] = average_scores(
                            scores_by_name(verdict.B_scores),
                            scores_by_name(reverse_verdict.A_scores))

                p.write(json.dumps(pipeline_summary, ensure_ascii=False) + '\n')
                s.write(json.dumps(singular_summary, ensure_ascii=False) + '\n')
                j.write(json.dumps(judge_summary, ensure_ascii=False) + '\n')
                p.flush(); s.flush(); j.flush()  # при сбое накопленные записи не теряются

        
if __name__ == "__main__":
    asyncio.run(main())