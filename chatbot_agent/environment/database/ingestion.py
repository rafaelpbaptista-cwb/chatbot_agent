"""Script contendo funções de ingestão de dados.

Segue indicação de ferramenta para testes da melhor configuração de spliter de dados: https://langchain-text-splitter.streamlit.app/
"""

import logging
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders.python import PythonLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def create_database() -> None:
    """Cria um banco de dados chromedb.

    O conteúdo desse banco de dados será usado na função RAG para acrescentar contexto
    as perguntas do usuário.
    """
    dir_database = os.getenv("DIR_DATABASE")
    dir_data_python = os.getenv("DIR_DATA_PYTHON")
    embedding_model = os.getenv("EMBEDDING_MODEL")

    logger.info("Variáveis utilizadas:")
    logger.info("DIR_DATABASE: %s", dir_database)
    logger.info("EMBEDDING_MODEL: %s", embedding_model)

    list_documents = _insert_doc_html_data(f"{dir_data_python}/docs/")
    list_documents.extend(_insert_python_data(dir_data_python))

    Chroma.from_documents(
        list_documents,
        GoogleGenerativeAIEmbeddings(model=embedding_model),
        persist_directory=os.getenv("DIR_DATABASE"),
    )


def _insert_doc_html_data(dir_docs_python: str) -> list[Document]:
    """Obtem informações da documentação Python a ser inserido no banco de dados.

    Args:
        dir_data_python (str):
            Diretório contendo a documentação Python a serem inseridos no banco
            de dados.

    Returns
    -------
        list[Document]: Lista de documentos.
    """
    loader = DirectoryLoader(
        dir_docs_python,
        glob="**/*.html",
        show_progress=True,
        use_multithreading=True,
        recursive=True,
        loader_cls=UnstructuredHTMLLoader,
    )

    list_documents = loader.load()
    for doc in list_documents:
        doc.metadata["type_data"] = "html"

    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.HTML,
        chunk_size=1000,
        chunk_overlap=500,
    )

    return splitter.split_documents(list_documents)


def _insert_python_data(dir_data_python: str) -> list[Document]:
    """Obtem informações de código Python a ser inserido no banco de dados.

    Args:
        dir_data_python (str):
            Diretório contendo os arquivos Python a serem inseridos no banco de dados.

    Returns
    -------
        list[Document]: Lista de documentos.
    """
    loader = DirectoryLoader(
        dir_data_python,
        glob="**/*.py",
        exclude=["**/__init__.py", "docs/**"],
        show_progress=True,
        use_multithreading=True,
        recursive=True,
        loader_cls=PythonLoader,
    )

    list_documents = loader.load()
    for doc in list_documents:
        doc.metadata["type_data"] = "python"

    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=4000,
        chunk_overlap=1000,
    )

    return splitter.split_documents(list_documents)
