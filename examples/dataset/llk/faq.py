import os
import uuid

import pandas as pd


def data_replace(path_data: str, request_col: str, answer_col: str) -> tuple:
    try:
        data = pd.read_excel(path_data)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None, None, None

    data[answer_col] = data[request_col] + "\n" + data[answer_col]  # TODO: удалить позже

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


def main(path_file: str, request_column: str, answer_column: str):
    corpus, queries, qrels = data_replace(path_file, request_column, answer_column)

    if corpus is not None:
        save_to_jsonl(corpus, "../../data/llk-faq/corpus.jsonl")

    if queries is not None:
        save_to_jsonl(queries, "../../data/llk-faq/queries.jsonl")

    if qrels is not None:
        save_to_tsv(qrels, "../../data/llk-faq/qrels", "test.tsv")


if __name__ == "__main__":
    PATH_FILE = "../../data/llk-faq/вопросы_faq_проверка.xlsx"
    REQUEST_COLUMN = "Вопрос"
    ANSWER_COLUMN = "Ответ"

    main(PATH_FILE, REQUEST_COLUMN, ANSWER_COLUMN)
