import os
import time
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask_model(prompt: str) -> tuple[str, float]:
    started = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Отвечай точно и по существу."},
            {"role": "user", "content": prompt},
        ],
    )

    elapsed = time.perf_counter() - started
    text = response.choices[0].message.content
    return text, elapsed


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_to_csv(results: list[dict], path: str | Path = OUTPUT_DIR / "results.csv") -> None:
    fieldnames = ["model", "prompt_id", "latency", "answer", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Результаты сохранены в {path}")


def save_to_json(results: list[dict], path: str | Path = OUTPUT_DIR / "results.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {path}")


if __name__ == "__main__":
    prompts = ["Объясни студенту 4 курса разницу между хешированием и симметричным шифрованием. Не более 120 слов.",
               "Определи тип события: «После 15 неудачных попыток входа с одного IP выполнен успешный вход администратора». Выбери: норма / подозрительно / критично и объясни.",
               "Сожми предложенный преподавателем текст до 5 тезисов. Промышленная революция XVIII–XIX веков стала переломным моментом в истории человечества. До её начала основой экономики повсеместно было сельское хозяйство и ручной труд. Однако с изобретением парового двигателя и механических ткацких станков ситуация кардинально изменилась. В Англии, а затем и в других странах Европы, начали появляться крупные фабрики, что привело к массовому перемещению людей из деревень в города. Это явление получило название урбанизации. С одной стороны, рост производства позволил значительно увеличить количество товаров и снизить их стоимость, что постепенно улучшило уровень жизни. С другой стороны, условия труда рабочих на ранних этапах были ужасающими: 14-часовой рабочий день, низкие зарплаты и использование детского труда. Кроме того, быстрый рост городов привел к антисанитарии и вспышкам эпидемий. Несмотря на эти тяжелые социальные издержки, именно промышленная революция заложила фундамент для современной экономики, науки и технологического прогресса, который мы наблюдаем сегодня.",
               "Извлеки из текста сущности и верни JSON с полями: ip, user, timestamp, event_type. 25 сентября 2026 года в 14:32 по московскому времени пользователь с логином ivan_petrov успешно авторизовался в системе с IP-адреса 192.168.1.45. Через минуту, в 14:33, с того же адреса поступил запрос на скачивание файла report_q3.pdf. Спустя час, в 15:35, пользователь ivan_petrov инициировал выход из системы (logout). Позже, в 16:10, с IP-адреса 10.0.0.77 была зафиксирована неудачная попытка входа под логином admin.",
               "Напиши Python-функцию проверки SHA-256 строки.",
               "Найди ошибку в предложенном фрагменте Python-кода и исправь её. def calculate_average(numbers): total = 0 for num in numbers: total += num average = total / len(numbers) return average my_list = [1, 2, 3, 4, 5] result = calculate_average(my_list) print(result)  empty_list = [] result2 = calculate_average(empty_list) print(result2)",
               "Предложи план первичного анализа подозрительного входа в корпоративную систему.",
               "Назови 5 рисков использования внешней LLM для анализа внутренних документов организации.",
               "Сформулируй краткое уведомление сотрудникам о запрете передачи паролей в AI-сервисы.",
               "Ответь только таблицей Markdown: риск | вероятность | ущерб | мера защиты. Оцени 4 риска AI-агента с доступом к почте."]

    results = []

    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] {prompt}")
        try:
            answer, latency = ask_model(prompt)
            results.append({
                "model": MODEL_NAME,
                "prompt_id": i,
                "latency": round(latency, 3),
                "answer": answer,
                "error": None,
            })
            print(f"    OK, {latency:.2f} c")
        except Exception as exc:
            results.append({
                "model": MODEL_NAME,
                "prompt_id": i,
                "latency": None,
                "answer": None,
                "error": str(exc),
            })
            print(f"    Ошибка: {exc}")

    save_to_csv(results)
    save_to_json(results)

    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    print(f"\nВсего запросов: {len(results)}, успешных: {len(ok)}, с ошибками: {len(failed)}")
    if ok:
        avg_latency = sum(r["latency"] for r in ok) / len(ok)
        print(f"Среднее время ответа: {avg_latency:.2f} c")