import logging
import os
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import ranx
import requests
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from ir_datasets.datasets.base import Dataset
from ir_datasets.formats import JsonlDocs, JsonlQueries, TrecQrels
from ir_datasets.util import LocalDownload
from pymilvus import CollectionSchema, DataType, MilvusClient
from pymilvus.milvus_client import IndexParams
from tqdm import tqdm

from examples.log_config import setup_logging

load_dotenv()

# --------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------
setup_logging("INFO")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
@dataclass
class Config:
    DATASET_NAME: str = "neon"
    DATASET_SUBSET: str = "neowiki"
    COLLECTION_NAME: str = "temp"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOCAL_DIR: Path = BASE_DIR / "data" / DATASET_NAME

    HF_ACCESS_TOKEN: str | None = os.getenv("HF_ACCESS_TOKEN")
    OPENAI_EMBEDDING_ENDPOINT: str | None = os.getenv("OPENAI_EMBEDDING_ENDPOINT")
    OPENAI_EMBEDDING_MODEL: str | None = os.getenv("OPENAI_EMBEDDING_MODEL")
    MILVUS_URI: str | None = os.getenv("MILVUS_URI")

    BATCH_SIZE: int = 128
    TOP_K: int = 1000

    @property
    def REPO_ID(self) -> str:
        return f"evg-kononov/{self.DATASET_NAME}"

    @cached_property
    def RUN_TAG(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.DATASET_NAME}_{self.DATASET_SUBSET}_{ts}"

    @property
    def RUN_OUTPUT_PATH(self) -> Path:
        return self.LOCAL_DIR / "runs" / f"{self.RUN_TAG}.tsv"

    def validate(self) -> None:
        missing = [
            name
            for name, value in [
                ("HF_ACCESS_TOKEN", self.HF_ACCESS_TOKEN),
                ("MILVUS_URI", self.MILVUS_URI),
                ("OPENAI_EMBEDDING_ENDPOINT", self.OPENAI_EMBEDDING_ENDPOINT),
                ("OPENAI_EMBEDDING_MODEL", self.OPENAI_EMBEDDING_MODEL),
            ]
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# --------------------------------------------------------------------
# Dataset Loading
# --------------------------------------------------------------------
def load_ir_dataset(docs_path: Path, queries_path: Path, qrels_path: Path) -> Dataset:
    docs = JsonlDocs(LocalDownload(docs_path))
    queries = JsonlQueries(LocalDownload(queries_path))
    qrels = TrecQrels(LocalDownload(qrels_path), {})
    return Dataset(docs, queries, qrels)


# --------------------------------------------------------------------
# Milvus Helpers
# --------------------------------------------------------------------
def create_schema(client: MilvusClient, fields: list[dict[str, Any]]) -> CollectionSchema:
    schema = client.create_schema()
    for field in fields:
        schema.add_field(**field)
    schema.verify()
    return schema


def create_index_params(client: MilvusClient, indexes: list[dict[str, Any]]) -> IndexParams:
    index_params = client.prepare_index_params()
    for index in indexes:
        index_params.add_index(**index)
    return index_params


def build_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    if client.has_collection(collection_name):
        logger.info("Collection '%s' already exists — deleting the collection.", collection_name)
        client.drop_collection(collection_name)

    fields = [
        {"field_name": "doc_id", "datatype": DataType.VARCHAR, "is_primary": True, "max_length": 64},
        {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": dim},
        # TODO: с хранением всего текста в Milvus возникают проблемы из-за ограничения на длину, использовать другую БД для хранения текста
        # {"field_name": "vector_text", "datatype": DataType.VARCHAR, "max_length": MAX_VECTOR_TEXT_LEN},
    ]
    indexes = [{"field_name": "vector", "index_type": "FLAT", "metric_type": "COSINE"}]

    schema = create_schema(client, fields)
    index_params = create_index_params(client, indexes)

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info("Created Milvus collection '%s' with dim=%d", collection_name, dim)


# --------------------------------------------------------------------
# Embedding API
# --------------------------------------------------------------------
def extract_embeddings(response_json: dict) -> list[list[float]]:
    return [item["embedding"] for item in response_json.get("data", [])]


def get_embeddings(input: list[str], url: str, model: str) -> list[list[float]]:
    # TODO: длина строки в input может быть больше, чем контекстное окно embedder'a, нужно решить эту проблему
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "input": input}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return extract_embeddings(response.json())
    except Exception as e:
        logger.warning("Embedding API error: %s", e)
        raise


# --------------------------------------------------------------------
# Batch Utilities
# --------------------------------------------------------------------
def batch_iterator(iterable: Iterable, batch_size: int) -> Generator[list[Any], None, None]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


# --------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------
def index_documents(
    client: MilvusClient,
    embed_url: str,
    embed_model: str,
    dataset: Dataset,
    collection_name: str,
    batch_size: int = 16,
) -> None:
    logger.info("Indexing documents into '%s' ...", collection_name)

    sample_vector = get_embeddings(["sample_text"], embed_url, embed_model)
    dim = len(sample_vector[0])
    build_collection(client, collection_name, dim)

    docs_iter = list(dataset.docs_iter())
    total_docs = len(docs_iter)

    for batch in tqdm(
        batch_iterator(docs_iter, batch_size),
        total=(total_docs // batch_size) + 1,
        desc="Indexing documents",
        unit="batch",
    ):
        texts = [doc.text for doc in batch]
        vectors = get_embeddings(texts, embed_url, embed_model)
        data = []
        for doc, vector in zip(batch, vectors, strict=False):
            data.append(
                {
                    "doc_id": doc.doc_id,
                    "vector": vector,
                }
            )

        client.insert(collection_name=collection_name, data=data)

    logger.info("Finished indexing %d documents.", total_docs)


# --------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------
@dataclass
class RetrievalResult:
    doc_id: str
    distance: float


def retrieve(
    client: MilvusClient,
    embed_url: str,
    embed_model: str,
    collection_name: str,
    top_k: int,
    queries: list[str],
) -> list[list[RetrievalResult]]:
    vectors = get_embeddings(queries, embed_url, embed_model)

    # TODO: правильно только для COSINE (сортирует от наибольшего к меньшему)
    search_results = client.search(collection_name=collection_name, data=vectors, limit=top_k, output_fields=["doc_id"])

    retrieval_results = []
    for results in search_results:
        retrieved = [RetrievalResult(doc_id=r["doc_id"], distance=r["distance"]) for r in results]
        retrieval_results.append(retrieved)

    return retrieval_results


# --------------------------------------------------------------------
# Run Generation
# --------------------------------------------------------------------
@dataclass
class TrecRunEntry:
    query_id: str
    doc_id: str
    rank: int
    score: int | float
    tag: str

    def to_trec_format(self) -> str:
        return f"{self.query_id}\tQ0\t{self.doc_id}\t{self.rank}\t{self.score}\t{self.tag}\n"


def generate_run(
    client: MilvusClient,
    embed_url: str,
    embed_model: str,
    dataset: Dataset,
    collection_name: str,
    output_path: Path,
    top_k: int,
    run_tag: str,
    batch_size: int = 16,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating run file: %s", output_path)

    queries = list(dataset.queries_iter())

    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(queries), batch_size), desc="Retrieving queries", unit="batch"):
            batch = queries[i : i + batch_size]
            batch_texts = [q.text for q in batch]
            batch_results = retrieve(client, embed_url, embed_model, collection_name, top_k, batch_texts)

            for q, results in zip(batch, batch_results, strict=False):
                for rank, r in enumerate(results, start=1):
                    entry = TrecRunEntry(q.query_id, r.doc_id, rank, r.distance, run_tag)
                    f.write(entry.to_trec_format())

    logger.info("Run file generated: %s", output_path)


# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------
def evaluate(qrels_path: Path, run_path: Path, metrics: list[str] | None = None) -> dict[str, float]:
    metrics = metrics or ["precision@10", "recall@10", "mrr@10", "map@10"]
    qrels = ranx.Qrels.from_file(str(qrels_path), kind="trec")
    run = ranx.Run.from_file(str(run_path), kind="trec")
    report = ranx.evaluate(qrels, run, metrics)
    logger.info("Evaluation results: %s", report)
    return report


# --------------------------------------------------------------------
# CLI Commands
# --------------------------------------------------------------------


def run_experiment(cfg: Config) -> dict[str, Any]:
    cfg.validate()
    client = MilvusClient(uri=cfg.MILVUS_URI)

    raw_data_dir = (
        Path(
            snapshot_download(
                repo_id=cfg.REPO_ID,
                local_dir=str(cfg.LOCAL_DIR),
                token=cfg.HF_ACCESS_TOKEN,
                allow_patterns=[f"{cfg.DATASET_SUBSET}/*"],
                repo_type="dataset",
            )
        )
        / cfg.DATASET_SUBSET
    )

    docs_path = raw_data_dir / "corpus.jsonl"
    queries_path = raw_data_dir / "queries.jsonl"
    qrels_path = raw_data_dir / "qrels" / "test.tsv"
    dataset = load_ir_dataset(docs_path, queries_path, qrels_path)

    index_documents(
        client,
        cfg.OPENAI_EMBEDDING_ENDPOINT,
        cfg.OPENAI_EMBEDDING_MODEL,
        dataset,
        cfg.COLLECTION_NAME,
        cfg.BATCH_SIZE,
    )
    generate_run(
        client,
        cfg.OPENAI_EMBEDDING_ENDPOINT,
        cfg.OPENAI_EMBEDDING_MODEL,
        dataset,
        cfg.COLLECTION_NAME,
        cfg.RUN_OUTPUT_PATH,
        cfg.TOP_K,
        cfg.RUN_TAG,
        cfg.BATCH_SIZE,
    )
    report = evaluate(qrels_path, cfg.RUN_OUTPUT_PATH)

    return {
        "dataset": f"{cfg.DATASET_NAME}/{cfg.DATASET_SUBSET}",
        "run_file": str(cfg.RUN_OUTPUT_PATH),
        **{k: float(v) for k, v in report.items()},
    }


def run():
    load_dotenv()
    setup_logging("INFO")
    cfg = Config()
    result = run_experiment(cfg)

    df = pd.DataFrame([result])
    df.to_excel("report.xlsx", index=False)


def run_all():
    load_dotenv()
    setup_logging("INFO")

    configs = [
        ("llk", "faq"),
        ("llk", "synthetic_faq"),
        ("neon", "faq"),
        ("neon", "neoportal"),
        ("neon", "neosite"),
        ("neon", "neostudy"),
        ("neon", "neowiki"),
    ]

    all_results = []
    for name, subset in configs:
        cfg = Config(DATASET_NAME=name, DATASET_SUBSET=subset)
        result = run_experiment(cfg)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_excel("report.xlsx", index=False)


if __name__ == "__main__":
    run_all()
