import pandas as pd
import json
from llama_cpp import Llama
from typing import List
from pydantic import BaseModel, ConfigDict
import time
import streamlit as st
from concat_headers import parse_csv, to_flat_csv
import re

class Triplet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str

class TripletsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    triplets: List[Triplet]

schema = TripletsPayload.model_json_schema()
print(schema)

SYSTEM_PROMPT = """
Ты — модель для извлечения триплетов из CSV-таблиц, где столбцы разделены ';'.

Верни только JSON:
{
  "triplets": [
    {"subject": "...", "predicate": "...", "object": "..."}
  ]
}

Извлекай только то, что явно есть в таблице. Ничего не придумывай.

ОСНОВНАЯ СХЕМА
- subject — сущность или параметр из первого столбца
- predicate — значение из ячейки
- object — заголовок колонки, к которой относится значение, или в основном, первая строка

НЕ ИЗВЛЕКАЙ
- пустые ячейки
- "-", "—", "–", null, NaN, None, Unnamed
- технические индексы вроде "0", "1", "2"

ТЕХНИЧЕСКИЕ ЭЛЕМЕНТЫ
- Игнорируй строки и колонки, состоящие только из технических индексов.
- Игнорируй служебный первый столбец с маркерами перечисления: "а)", "б)", "в)", "1)", "2)", "a)", "b)", "№".
- Если верхняя строка состоит из "0;1;2;..." — это не данные.
- Если заголовок пустой object = "".

ТИПЫ ТАБЛИЦ

1. ОБЫЧНАЯ МАТРИЦА
Если есть строка заголовков и первый осмысленный столбец с названиями строк:
- subject = значение из первого осмысленного столбца
- predicate = значение ячейки
- object = заголовок колонки

Пример:
№ п/п;Сталь;Cu;Mg
1;ABC;0.1;1.2
2;type2;1.1;0-0.1
3;type1;-;0.01

Результат:
{
  "triplets": [
    {"subject": "ABC", "predicate": "0.1", "object": "Cu"},
    {"subject": "ABC", "predicate": "1.2", "object": "Mg"},
    {"subject": "type2", "predicate": "1.1", "object": "Cu"},
    {"subject": "type2", "predicate": "0-0.1", "object": "Mg"},
    {"subject": "type1", "predicate": "0.01", "object": "Mg"}
  ]
}

2. ПАРАМЕТР → ЗНАЧЕНИЕ
Если таблица по смыслу является списком параметров и значений, а не матрицей:
- horizontal: одна строка заголовков + одна строка значений (может включать в себя индексы или буквенными перечислениями)
  subject = название колонки
  predicate = значение
  object = ""
- vertical: первый столбец = параметр, второй = значение
  subject = первый столбец
  predicate = второй столбец
  object = ""

3. ОБЪЕКТ + ХАРАКТЕРИСТИКИ
Если первая осмысленная колонка содержит сущность, а остальные — её свойства:
- subject = значение из первой осмысленной колонки
- predicate = значение соседней ячейки
- object = заголовок второй и далее колонок

Пример:
0;Эффект ниобия;Состояние ниобия
1;Измельчение зерна аустенита;NbCy
2;Торможение рекристаллизации;NbC, [Nb]y

Результат:
{
  "triplets": [
    {"subject": "Измельчение зерна аустенита", "predicate": "NbCy", "object": "Состояние ниобия"},
    {"subject": "Торможение рекристаллизации", "predicate": "NbC, [Nb]y", "object": "Состояние ниобия"}
  ]
}

4. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
Если первая строка содержит название класса объектов в первом столбце и названия объектов в остальных столбцах,
а ниже в первом столбце идут параметры, то:
- первая строка — это header, не данные
- subject = параметр из первого столбца
- predicate = значение в ячейке
- object = название объекта из первой строки

Пример:
Агломашина;К-3-50;К-3-75
Высота слоя шихты, мм;200-;250
None;250;260
Вертикальная скорость спекания, мм/мин;28-35;33-35

Правильная интерпретация:
- "Агломашина" — это заголовок первого столбца, не subject
- "К-3-50" и "К-3-75" — это object
- строка с пустым первым столбцом может продолжать предыдущую строку

Если строка с пустым первым столбцом продолжает предыдущее значение, объединяй их:
- "200-" + "250" → "200-250"
- аналогично для других колонок той же строки

Результат:
{
  "triplets": [
    {"subject": "Высота слоя шихты, мм", "predicate": "200-250", "object": "К-3-50"},
    {"subject": "Высота слоя шихты, мм", "predicate": "250-260", "object": "К-3-75"},
    {"subject": "Вертикальная скорость спекания, мм/мин", "predicate": "28-35", "object": "К-3-50"},
    {"subject": "Вертикальная скорость спекания, мм/мин", "predicate": "33-35", "object": "К-3-75"}
  ]
}

ПРАВИЛА
- Не делай object равным subject, если в таблице есть отдельные заголовки колонок.
- Не извлекай строки header как triplets.
- Не используй служебные метки и технические индексы как subject или object.
- Если ячейка длинная текстовая, возвращай её целиком.
- Если заголовок отсутствует или неосмыслен, object = "".
- Если предикат отсутсвует или поставлен как "-"/".", игнорируй такой триплет.
- В случае, если объект или субъект представлены цифрами, например, столбец "Номер режима", в котором значения "1"/"2", то соединяй такие значения с названием столбца, по примеру "Режим 1"/"Режим 2"


Верни только JSON без пояснений и без markdown.
"""

model = Llama.from_pretrained(
    repo_id="t-tech/T-lite-it-2.1-GGUF",
    filename="*Q5_K_M.gguf",
    verbose=True,
    n_ctx=4096
)

# model = Llama.from_pretrained(
#     repo_id="Qwen/Qwen3-8B-GGUF",
#     filename="*Q5_K_M.gguf",
#     verbose=True,
#     n_ctx=4096
# )

def has_only_technical_headers(df) -> bool:
    cols = [str(c).strip() for c in df.columns]
    return cols == [str(i) for i in range(len(cols))]

def table_to_csv_text(table, sep: str = ";") -> str:
    df = table.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    df = df.map(
        lambda x: x.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        if isinstance(x, str) else x
    )

    df.columns = [
        (
            str(c).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        )
        for c in df.columns
    ]

    write_header = not has_only_technical_headers(df)
    csv_text = df.to_csv(index=False, header=write_header, sep=sep)

    headers, datarows = parse_csv(csv_text)
    flat_csv = to_flat_csv(headers, datarows)

    lines = flat_csv.split("\n")
    lines[0] = re.sub(r"-\s+", "", lines[0])
    flat_csv = "\n".join(lines)

    print(flat_csv)
    return flat_csv

def create_user_prompt_for_table(table):
    csv_table = table_to_csv_text(table, sep=";")
    return csv_table

@st.cache_resource
def extract_triplets_for_one_table(table):
    user_prompt = create_user_prompt_for_table(table)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    start = time.time()
    response = model.create_chat_completion(
        messages,
        response_format={
            "type": "json_object",
            "schema": schema
        },
        temperature=0.3,
        top_p=0.8,
        top_k=50,
        max_tokens=4096
    )
    end = time.time()

    print(f"ВРЕМЯ ВЫПОЛНЕНИЯ ЗАПРОСА: {end - start:.2f} сек")
    print(f"ПОЛЬЗОВАТЕЛЬСКИЙ ЗАПРОС:\n{user_prompt}")

    outputs = response["choices"][0]["message"]["content"]

    try:
        return json.loads(outputs)
    except Exception:
        return outputs

@st.cache_resource
def extract_triplets_by_llm(pandas_table):
    result = extract_triplets_for_one_table(pandas_table)

    return result
