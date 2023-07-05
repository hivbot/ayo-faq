import util as util
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AwaDB, Chroma
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
import os
import shutil
from enum import Enum

EMBEDDING_MODEL_FOLDER = ".embedding-model"
VECTORDB_FOLDER = ".vectordb"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
VECTORDB_TYPES = Enum("VECTORDB_TYPES", ["AwaDB", "Chroma"])
VECTORDB_TYPE = VECTORDB_TYPES.AwaDB


def create_documents(df: pd.DataFrame, page_content_column: str) -> pd.DataFrame:
    loader = DataFrameLoader(df, page_content_column=page_content_column)
    return loader.load()


def define_embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=EMBEDDING_MODEL_FOLDER,
    )


def get_vectordb(
    faq_id: str,
    embedding_function: Embeddings,
    documents: List[Document] = None,
    vectordb_type: str = VECTORDB_TYPE,
) -> VectorStore:
    vectordb = None

    if vectordb_type is VECTORDB_TYPES.AwaDB:
        if documents is None:
            vectordb = AwaDB(
                embedding=embedding_function, log_and_data_dir=VECTORDB_FOLDER
            )
            if not vectordb.load_local(table_name=faq_id):
                raise Exception("faq_id may not exists")
        else:
            vectordb = AwaDB.from_documents(
                documents=documents,
                embedding=embedding_function,
                table_name=faq_id,
                log_and_data_dir=VECTORDB_FOLDER,
            )
    if vectordb_type is VECTORDB_TYPES.Chroma:
        if documents is None:
            vectordb = Chroma(
                collection_name=faq_id,
                embedding_function=embedding_function,
                persist_directory=VECTORDB_FOLDER,
            )
            if not vectordb.get()["ids"]:
                raise Exception("faq_id may not exists")
        else:
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=faq_id,
                persist_directory=VECTORDB_FOLDER,
            )
    return vectordb


def similarity_search(
    vectordb: VectorStore, query: str, k: int = 3
) -> List[Tuple[Document, float]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return vectordb.similarity_search_with_relevance_scores(query=query, k=k)


def load_vectordb_id(
    faq_id: str,
    page_content_column: str,
    embedding_function_name: str = EMBEDDING_MODEL,
) -> VectorStore:
    embedding_function = define_embedding_function(embedding_function_name)
    vectordb = None
    try:
        vectordb = get_vectordb(faq_id=faq_id, embedding_function=embedding_function)
    except Exception as e:
        print(e)
        vectordb = create_vectordb_id(faq_id, page_content_column, embedding_function)

    return vectordb


def create_vectordb_id(
    faq_id: str,
    page_content_column: str,
    embedding_function: HuggingFaceEmbeddings = None,
) -> VectorStore:
    if embedding_function is None:
        embedding_function = define_embedding_function(EMBEDDING_MODEL)

    df = util.read_df(util.xlsx_url(faq_id))
    documents = create_documents(df, page_content_column)
    vectordb = get_vectordb(
        faq_id=faq_id, embedding_function=embedding_function, documents=documents
    )
    return vectordb


def load_vectordb(sheet_url: str, page_content_column: str) -> VectorStore:
    return load_vectordb_id(util.get_id(sheet_url), page_content_column)


def delete_vectordb():
    shutil.rmtree(VECTORDB_FOLDER, ignore_errors=True)
