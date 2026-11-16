import asyncio
import os, logging, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import get_usage_metadata_callback
from orchestrator import    (TaskState,
                            format_status_report,
                            format_usage_report,
                            format_feedback,
                            create_app, )
from judge import cast_judgement

load_dotenv()
# Единая настройка логирования для всего приложения
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # убрать шум HTTP-запросов
logging.getLogger("httpx2").setLevel(logging.WARNING)


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

    with    open('tasks.jsonl', 'r', encoding='utf-8') as f, 
            open('pipeline_results.jsonl', 'w+', encoding='utf-8') as p,
            open('singular_results.jsonl', 'w+', encoding='utf-8') as s,
            open('judge_summary.jsonl', 'w+', encoding='utf-8') as j:

        for line in f:
            if not line.strip():
                continue
            line_dict = json.loads(line)
            task = line_dict["task"]

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

            final_answer_A = ""
            task_failed = False
            technical_fail = False
            model_usage = {}
            try:
                with get_usage_metadata_callback() as usage_cb:
                    result = await orchestrator_app.ainvoke(initial_state)
                
                if result.get("final_answer"):
                    final_answer_A = result["final_answer"]
                else:
                    task_failed = True
                revision_count = result['revision_count']
                for model, usage in usage_cb.usage_metadata.items():
                    inp = usage.get("input_tokens", 0)
                    out = usage.get("output_tokens", 0)
                    model_usage[model] = {"input": inp, "output": out}
            except Exception as e:
                technical_fail = True
            
            pipeline_summary = { 
                "id": task["id"], 
                "no_response": technical_fail, 
                "failed": task_failed,
                "rev_count": revision_count,
                "model_usage": model_usage
            }

            # --------------------------------------------------------------------------------------

            usage_cb = None

            final_answer_B = ""
            task_failed = False
            technical_fail = False
            model_usage = {}
            try:
                with get_usage_metadata_callback() as usage_cb:
                    result = await singular_llm.ainvoke(task)
                
                if result:
                    final_answer_B = result
                else:
                    task_failed = True
                
                for model, usage in usage_cb.usage_metadata.items():
                    inp = usage.get("input_tokens", 0)
                    out = usage.get("output_tokens", 0)
                    model_usage[model] = {"input": inp, "output": out}
            except Exception as e:
                technical_fail = True
            
            singular_summary = { 
                "id": task["id"], 
                "no_response": technical_fail, 
                "failed": task_failed,
                "model_usage": model_usage
            }

            ####
            verdict =           await cast_judgement(task, final_answer_A, final_answer_B)
            reverse_verdict =   await cast_judgement(task, final_answer_B, final_answer_A)

            judge_summary = {
                "id": task["id"],
                "winner":       {"fwd": verdict.winner,
                                "bwd": reverse_verdict.winner},
                "confidence":   {"fwd": verdict.confidence,
                                "bwd": reverse_verdict.confidence}
            }

            pipeline_summary["score"] = (verdict.A_score + reverse_verdict.B_score) / 2
            singular_summary["score"] = (verdict.B_score + reverse_verdict.A_score) / 2

            p.write(json.dumps(pipeline_summary) + '\n')
            s.write(json.dumps(singular_summary) + '\n')
            j.write(json.dumps(judge_summary) + '\n')

        

if __name__ == "__main__":
    asyncio.run(main())
