"""Script contendo definição dos documentos estruturados.

Documentos aqui contidos são usados nas resposta dos chains e
outros guardam o estado do aplicação Graph.
"""

from typing import TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class DocumentsGraderAnswer(BaseModel):
    """Classe estruturada usada para retornar resposta do RagGrader."""

    is_relevante: bool = Field(description="Nota de relevância podendo ser 0 ou 1")
    explaination: str = Field(
        description="Explicação (máximo 100 caracteres) sucinta do score"
    )
    analized_document: Document = Field(description="Documento Analisado")


class GraphState(TypedDict):
    """Classe que representa o estado na nossa aplicação (Graph)."""

    question: str
    response: str
    documents: list[str]
