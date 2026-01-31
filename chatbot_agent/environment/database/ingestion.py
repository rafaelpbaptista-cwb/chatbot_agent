"""Script contendo funções de ingestão de dados.

Segue indicação de ferramenta para testes da melhor configuração de spliter de dados: https://langchain-text-splitter.streamlit.app/
"""

import logging
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.python import PythonLoader
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

    loader = DirectoryLoader(
        dir_data_python,
        glob="**/*.py",
        exclude=["**/__init__.py"],
        show_progress=True,
        use_multithreading=True,
        recursive=True,
        loader_cls=PythonLoader,
    )

    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=4000,
        chunk_overlap=1000,
    )

    Chroma.from_documents(
        splitter.split_documents(loader.load()),
        GoogleGenerativeAIEmbeddings(model=embedding_model),
        persist_directory=os.getenv("DIR_DATABASE"),
    )
