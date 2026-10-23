import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUTPUT_DIR / "result.json"


def load_results(path: Path = JSON_PATH) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_to_json(results: list[dict], path: Path = JSON_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {path}")


def ask_model(prompt: str,
    *,
    experiment: str = "default",
    system: str | None = None,
    system_id: str | None = None,
    temperature=None,
    max_tokens=None,
    repeat: int = 1,
):
    results = load_results()
    ok, failed = 0, 0
    latencies = []

    for r in range(1, repeat + 1):
        print(f"[{r}/{repeat}] {prompt}")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": MODEL_NAME, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        started = time.perf_counter()
        usage_dict = None
        try:
            response = client.chat.completions.create(**kwargs)
            answer = response.choices[0].message.content
            error = None

            usage = getattr(response, "usage", None)
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
        except Exception as e:
            answer = None
            error = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - started

        if error is None:
            ok += 1
            latencies.append(latency)
            tokens_info = (
                f", tokens: {usage_dict['total_tokens']}"
                if usage_dict and usage_dict.get("total_tokens") is not None
                else ""
            )
            print(f"    OK, {latency:.2f} c{tokens_info}")
        else:
            failed += 1
            print(f"    Ошибка: {error}")

        results.append({
            "model": MODEL_NAME,
            "experiment": experiment,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_id": system_id,
            "system_prompt": system,
            "repeat": r,
            "latency_s": round(latency, 3),
            "answer": answer,
            "usage": usage_dict,
            "error": error,
        })

    print(f"\nВсего запросов: {repeat}, успешных: {ok}, с ошибками: {failed}")

    save_to_json(results)

if __name__ == "__main__":
    ask_model(
        prompt = "Объясни студенту 4 курса архитектуру RAG-системы: ingestion, chunking, embeddings, vector database, retrieval, prompt construction и generation.",
        system_id = "sys_00",
        experiment = "prompt_limits",
        max_tokens = 1600,
        repeat =  1
    )