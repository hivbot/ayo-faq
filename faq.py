import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AwaDB
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
import os

SHEET_URL_X = "https://docs.google.com/spreadsheets/d/"
SHEET_URL_Y = "/edit#gid="
SHEET_URL_Y_EXPORT = "/export?gid="
CACHE_FOLDER = ".embedding-model"
VECTORDB_FOLDER = ".vectordb"


def faq_id(sheet_url: str) -> str:
    x = sheet_url.find(SHEET_URL_X)
    y = sheet_url.find(SHEET_URL_Y)
    return sheet_url[x + len(SHEET_URL_X) : y] + "-" + sheet_url[y + len(SHEET_URL_Y) :]


def xlsx_url(faq_id: str) -> str:
    y = faq_id.rfind("-")
    return SHEET_URL_X + faq_id[0:y] + SHEET_URL_Y_EXPORT + faq_id[y + 1 :]


def read_df(xlsx_url: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_url, header=0, keep_default_na=False)


def create_documents(df: pd.DataFrame, page_content_column: str) -> pd.DataFrame:
    loader = DataFrameLoader(df, page_content_column=page_content_column)
    return loader.load()


def define_embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=CACHE_FOLDER,
    )


def get_vectordb(
    faq_id: str, embedding_function: Embeddings, documents: List[Document] = None
) -> VectorStore:
    vectordb = None
    if documents is None:
        vectordb = AwaDB(embedding=embedding_function, log_and_data_dir=VECTORDB_FOLDER)
        success = vectordb.load_local(table_name=faq_id)
        if not success:
            raise Exception("faq_id may not exists")
    else:
        vectordb = AwaDB.from_documents(
            documents=documents,
            embedding=embedding_function,
            table_name=faq_id,
            log_and_data_dir=VECTORDB_FOLDER,
        )
    return vectordb


def similarity_search(
    vectordb: VectorStore, query: str, k: int = 3
) -> List[Tuple[Document, float]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return vectordb.similarity_search_with_relevance_scores(query=query, k=k)


def load_vectordb_id(faq_id: str, page_content_column: str) -> VectorStore:
    embedding_function = define_embedding_function("sentence-transformers/all-mpnet-base-v2")
    vectordb = None
    try:
        vectordb = get_vectordb(faq_id=faq_id, embedding_function=embedding_function)
    except Exception as e:
        df = read_df(xlsx_url(faq_id))
        documents = create_documents(df, page_content_column)
        vectordb = get_vectordb(faq_id=faq_id, embedding_function=embedding_function, documents=documents)
    return vectordb


def load_vectordb(sheet_url: str, page_content_column: str) -> VectorStore:
    return load_vectordb_id(faq_id(sheet_url), page_content_column)
