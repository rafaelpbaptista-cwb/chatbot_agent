"""Script contendo definição dos documentos estruturados.

Documentos aqui contidos são usados nas resposta dos chains e
outros guardam o estado do aplicação Graph.
"""

from typing import TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class DocumentsGraderAnswer(BaseModel):
    """Classe estruturada usada para retornar resposta do RagGrader."""

    Score: int = Field(description="Relevance score between 0 and 1")
    Explaination: str = Field(description="Explanation of relevance score")
    analized_document: Document = Field(description="Analized Document")


class GraphState(TypedDict):
    """Classe que representa o estado na nossa aplicação (Graph)."""

    question: str
    response: str
    generation: str
    documents: list[str]
