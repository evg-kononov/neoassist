import json
import os
import uuid

import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

PROMPT_EN = """You're an assistant in composing questions in Russian. 
Your task is to change the QUESTION you asked, and the prerequisite is to have an answer to the question \
you wrote in ANSWER.
The meaning of the question should not change. When changing a question, you can use synonyms, swap words, \
use other styles (colloquial, formal, etc.).
### Work examples:
1. QUESTION: "Где можно найти информацию о топливе и откуда оно поставляется на АЗС Teboil?"
ANSWER: "Высококачественный бензин и дизельное топливо поставляются на АЗС Teboil организациями группы «ЛУКОЙЛ», \
а также проверенными партнёрами. 
В уголке потребителя каждой АЗС Teboil доступны паспорта качества топлива с информацией о поставщике и \
характеристиках топлива."
Assistant: {{"rephrase_query": Топливо на АЗС Teboil. Откуда и кто поставляет?}}
2. QUESTION: "Масло с каким классом вязкости лучше использовать зимой?"
ANSWER: "Масло с каким классом вязкости лучше использовать зимой?"
ОТВЕТ: "Важно, чтобы при низких температурах масло сохраняло такую вязкость, которая позволит ему \
прокачаться по каналам системы смазки, и не будет мешать проворачиваемости коленчатого вала. Поэтому есть \
разбивка на «зимние» классы по SAE J300 и в зависимости от минимальных температур применения масла могут \
рекомендоваться продукты различных «зимних» классов вязкости.
У всесезонного масла кинематическая вязкость при 100 оС не зависит от того, какой это продукт ""0W-30"" или \
""5W-30"". На кинематическую вязкость при рабочих температурах влияет, как раз, «летний» класс без W, то есть SAE«30».
Поэтому кинематическая вязкость при 100 оС масел классов 0W-30 и 5W-30 может быть одинакова. И масло 0W-30 при \
рабочих температурах не будет более жидким, чем 5W-30.
С более подробной информацией о вязкости продукта можно ознакомиться по ссылке: \
https://teboil.ru/articles/vyazkost-motornykh-masel/ 
Современные синтетические моторные масла классов вязкости 0W-20, 5W-40, 5W-30, 0W-30 предназначены для всесезонного \
применения, и использовать их возможно и необходимо как летом, так и зимой."
Assistant: {{"rephrase_query": Какие классы вязкости подходят для зимней эксплуатации?}}
3. QUESTION: "Продавец "Торговая сеть Подкова""	
ANSWER: "Указанный магазин/сеть является официальной точкой продажи смазочных материалов Teboil."
Assistant: {{"rephrase_query": "Торговая сеть Подкова" является ли официальной точкой продажи Teboil или нет?}}
4. QUESTION: "Что такое Carbon-to-lubes?"
ANSWER: "Carbon-to-Lubes – это синергия кристально чистых базовых масел, премиального 
пакета адаптивных присадок и технологии производства. «Синергия» означает максимально 
эффективное взаимодействие основных компонентов в условиях высокотехнологичного 
способа производства продукта, а именно: высокоточного потокового смешения. Передовая 
технология синтеза моторного масла обеспечивает длинные прочные молекулярные связи, 
что позволяет ему сохранять стабильность в любых условиях эксплуатации."
Assistant: {{"rephrase_query": "Carbon to lubes че это такое?"}}
5. QUESTION: "Какие есть вакансии в Teboil?"	
ANSWER: "О том, как начать карьеру в АЗС Teboil, Вы можете прочитать на официальном сайте: \
https://azs.teboil.ru/career/"
Assistant: {{"rephrase_query": "где узнать как начать карьеру в азс teboil"}}

### Let's get started:
QUESTION: {question}
ANSWER: {answer}
"""


class RephraseQuerySO(BaseModel):
    rephrase_query: str = Field(description="""The changed text of the question""")


def llm_initialization(
    base_url: str, api_key: str, model_name: str, path_tokenizer: str, temp: float, scheme: type[BaseModel]
) -> ChatOpenAI:
    with open(path_tokenizer) as f:
        config = json.load(f)

    chat_template = config["chat_template"]

    model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temp,
        extra_body={"chat_template": chat_template},
    )
    return model.with_structured_output(scheme)


def call_llm(query: str, answer: str, model: ChatOpenAI, prompt: str) -> RephraseQuerySO:
    system_message = SystemMessage(content=prompt.format(question=query, answer=answer))
    output_so = model.invoke([system_message])
    return output_so


def try_value(x: RephraseQuerySO) -> str | None:
    try:
        return list(vars(x).values())[0]  # Попытка получить значение
    except Exception:
        return None  # Обработка исключения


def generation_data(
    data: pd.DataFrame, llm: ChatOpenAI, request_column: str, answer_column: str, prompt: str
) -> pd.DataFrame:
    print("Генерация вопросов...")
    tqdm.pandas()
    data["generated_SO"] = data.progress_apply(
        lambda row: call_llm(query=row[request_column], answer=row[answer_column], model=llm, prompt=prompt), axis=1
    )
    data["generated_query"] = data["generated_SO"].apply(lambda x: (try_value(x)) if x is not None else None)
    data = data[data["generated_query"].notna()]

    return data


def dataframe_replace(data: pd.DataFrame, request_col: str, answer_col: str) -> tuple:
    data_query = pd.DataFrame()
    data_query[request_col] = data[request_col].drop_duplicates()
    data_query["query_id"] = [str(uuid.uuid4()) for _ in range(len(data_query))]

    data_answer = pd.DataFrame()
    data_answer[answer_col] = data[answer_col].drop_duplicates()
    data_answer["doc_id"] = [str(uuid.uuid4()) for _ in range(len(data_answer))]

    data_new = data.merge(data_query, how="left", on=request_col)
    data_new = data_new.merge(data_answer, how="left", on=answer_col)
    data_new["score"] = 1

    queries = data_query.rename(columns={request_col: "text"})
    corpus = data_answer.rename(columns={answer_col: "text"})
    qrels = data_new[["query_id", "doc_id", "score"]].drop_duplicates()
    qrels.insert(1, "iteration", 0)

    return corpus, queries, qrels


def save_to_jsonl(dataframe: pd.DataFrame, filename: str):
    dataframe.to_json(filename, orient="records", lines=True, force_ascii=False)
    print(f"Данные сохранены в {filename}")


def save_to_tsv(dataframe: pd.DataFrame, directory: str, filename: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    dataframe.to_csv(file_path, sep="\t", index=False, header=False)
    print(f"Данные сохранены в {file_path}")


def main(
    path_file: str,
    request_column: str,
    answer_column: str,
    base_url: str,
    api_key: str,
    model_name: str,
    path_tokenizer: str,
    temp: float,
    scheme: type[BaseModel],
    prompt: str,
):
    model = llm_initialization(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        path_tokenizer=path_tokenizer,
        temp=temp,
        scheme=scheme,
    )

    try:
        data = pd.read_excel(path_file)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

    data_gen = generation_data(
        data=data, llm=model, request_column=request_column, answer_column=answer_column, prompt=prompt
    )
    corpus, queries, qrels = dataframe_replace(data_gen, request_col="generated_query", answer_col=answer_column)
    if corpus is not None:
        save_to_jsonl(corpus, "corpus.jsonl")

    if queries is not None:
        save_to_jsonl(queries, "queries.jsonl")

    if qrels is not None:
        save_to_tsv(qrels, "qrels", "test.tsv")


if __name__ == "__main__":
    PATH_FILE = "вопросы_faq_проверка.xlsx"
    REQUEST_COLUMN = "Вопрос"
    ANSWER_COLUMN = "Ответ"

    OPENAI_BASE_URL = "http://87.242.104.103:1234/v1"
    OPENAI_API_KEY = "some_key"
    CHAT_MODEL_NAME = "qwen2:72b"
    TOKENIZER_PATH = "../tokenizer_config.json"
    TEMPERATURE = 0.5

    main(
        path_file=PATH_FILE,
        request_column=REQUEST_COLUMN,
        answer_column=ANSWER_COLUMN,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_name=CHAT_MODEL_NAME,
        path_tokenizer=TOKENIZER_PATH,
        temp=TEMPERATURE,
        scheme=RephraseQuerySO,
        prompt=PROMPT_EN,
    )
