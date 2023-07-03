import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AwaDB
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
import os

sheet_url_x = "https://docs.google.com/spreadsheets/d/"
sheet_url_y = "/edit#gid="
sheet_url_y_exp = "/export?gid="
cache_folder=".embedding-model"
dir_vectordb = ".vectordb"


def faq_id(sheet_url: str) -> str:
    x = sheet_url.find(sheet_url_x)
    y = sheet_url.find(sheet_url_y)
    return sheet_url[x + len(sheet_url_x) : y] + "-" + sheet_url[y + len(sheet_url_y) :]


def xlsx_url(faq_id: str) -> str:
    y = faq_id.rfind("-")
    return sheet_url_x + faq_id[0:y] + sheet_url_y_exp + faq_id[y + 1 :]


def read_df(xlsx_url: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_url, header=0, keep_default_na=False)


def create_documents(df: pd.DataFrame, page_content_column: str) -> pd.DataFrame:
    loader = DataFrameLoader(df, page_content_column=page_content_column)
    return loader.load()


def embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=cache_folder
    )


def vectordb(
    faq_id: str,
    embedding_function: Embeddings,
    documents: List[Document] = None
) -> VectorStore:
    vectordb = None
    if documents is None:
        vectordb = AwaDB(
            embedding=embedding_function,
            log_and_data_dir=dir_vectordb
        )
        success = vectordb.load_local(table_name=faq_id)
        if not success:
            raise Exception("faq_id may not exists")
    else:
        vectordb = AwaDB.from_documents(
            documents=documents,
            embedding=embedding_function,
            table_name=faq_id,
            log_and_data_dir=dir_vectordb
        )
    return vectordb


def similarity_search(vectordb: VectorStore, query: str, k: int) -> List[Tuple[Document, float]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return vectordb.similarity_search_with_relevance_scores(query=query, k=k)