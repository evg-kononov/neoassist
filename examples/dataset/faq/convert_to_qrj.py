# TODO: подумать над разбиением на train/dev/test (dev/test, если без обучения модели) и стратификацией

from pathlib import Path
from uuid import UUID

import pandas as pd
import pandera.pandas as pa


def is_uuid(s: pd.Series) -> pd.Series:
    def ok(v: str) -> bool:
        try:
            UUID(v)
            return True
        except Exception:
            return False

    return s.map(ok)


uuid_check = pa.Check(is_uuid, name="is_uuid", error="Значение должно быть корректным UUID")

input_corpus_schema = pa.DataFrameSchema(
    {
        "doc_id": pa.Column(pa.String, unique=True, checks=uuid_check),
        "question": pa.Column(pa.String, checks=pa.Check.str_length(1)),
        "answer": pa.Column(pa.String, checks=pa.Check.str_length(1)),
    },
    coerce=True,
    strict=True,
)

input_queries_schema = pa.DataFrameSchema(
    {
        "query_id": pa.Column(pa.String, checks=uuid_check),
        "text": pa.Column(pa.String, checks=pa.Check.str_length(1)),
        "doc_id": pa.Column(pa.String, checks=uuid_check),
        "score": pa.Column(pa.Int, checks=[pa.Check.ge(0), pa.Check.le(3)]),
    },
    coerce=True,
    strict=True,
)

output_corpus_schema = pa.DataFrameSchema(
    {
        "doc_id": pa.Column(pa.String, unique=True, checks=uuid_check),
        "text": pa.Column(pa.String, checks=pa.Check.str_length(1)),
    },
    coerce=True,
    strict=True,
)

output_queries_schema = pa.DataFrameSchema(
    {
        "query_id": pa.Column(pa.String, unique=True, checks=uuid_check),
        "text": pa.Column(pa.String, checks=pa.Check.str_length(1)),
    },
    coerce=True,
    strict=True,
)

output_qrels_schema = pa.DataFrameSchema(
    {
        "query_id": pa.Column(pa.String, checks=uuid_check),
        "iteration": pa.Column(pa.Int, checks=pa.Check.eq(0)),
        "doc_id": pa.Column(pa.String, checks=uuid_check),
        "score": pa.Column(pa.Int, checks=[pa.Check.ge(0), pa.Check.le(3)]),
    },
    coerce=True,
    strict=True,
)


@pa.check_io(corpus=output_corpus_schema, queries=output_queries_schema, qrels=output_qrels_schema)
def assert_referential(corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame) -> None:
    missing_docs = set(qrels["doc_id"]) - set(corpus["doc_id"])
    missing_qids = set(qrels["query_id"]) - set(queries["query_id"])
    if missing_docs or missing_qids:
        raise ValueError(
            f"Нарушение ссылочной целостности qrels: "
            f"{len(missing_docs)} отсутствующих doc_ids, {len(missing_qids)} отсутствующих query_ids"
        )

@pa.check_output(input_corpus_schema)
def load_corpus(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


@pa.check_output(input_queries_schema)
def load_queries(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"doc_id": df["doc_id"], "text": df["question"] + "\n" + df["answer"]}).sort_values("doc_id")


def build_queries(df: pd.DataFrame) -> pd.DataFrame:
    return df[["query_id", "text"]].drop_duplicates(subset="query_id").sort_values("query_id").reset_index(drop=True)


def build_qrels(df: pd.DataFrame) -> pd.DataFrame:
    qrels = pd.DataFrame(
        {"query_id": df["query_id"], "iteration": 0, "doc_id": df["doc_id"], "score": df["score"]}
    ).drop_duplicates().sort_values(["query_id", "doc_id"]).reset_index(drop=True)
    return qrels


@pa.check_input(output_corpus_schema)
def save_corpus(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", lines=True, force_ascii=False)


@pa.check_input(output_queries_schema)
def save_queries(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", lines=True, force_ascii=False)


@pa.check_input(output_qrels_schema)
def save_qrels(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, header=False)


def main() -> None:
    corpus_path = Path("./corpus.xlsx")
    queries_path = Path("./queries.xlsx")
    out_dir = Path("./output")

    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_df = load_corpus(corpus_path)
    queries_df = load_queries(queries_path)

    corpus = build_corpus(corpus_df)
    queries = build_queries(queries_df)
    qrels = build_qrels(queries_df)

    assert_referential(corpus, queries, qrels)

    save_corpus(corpus, out_dir / "corpus.jsonl")
    save_queries(queries, out_dir / "queries.jsonl")
    save_qrels(qrels, out_dir / "qrels.tsv")


if __name__ == "__main__":
    main()
