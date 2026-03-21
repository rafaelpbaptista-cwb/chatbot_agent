"""Pacote principal do projeto."""

from dotenv import load_dotenv

from .chain.create_chains import (
    create_documents_html_grader,
    create_generate,
    create_query_retriever,
)
from .consts import GENERATE, GRADE_DOCUMENTS, RETRIEVER_HTML
from .structured_output.models import GraphState

load_dotenv()


__all__ = [
    "GENERATE",
    "GRADE_DOCUMENTS",
    "RETRIEVER_HTML",
    "GraphState",
    "create_documents_html_grader",
    "create_generate",
    "create_query_retriever",
]
