"""Script contendo definição dos documentos estruturados.

Documentos aqui contidos são usados nas resposta dos chains e
outros guardam o estado do aplicação Graph.
"""

from langchain_classic.schema import BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class DocumentsGraderAnswer(BaseModel):
    """Classe estruturada usada para retornar resposta do RagGrader.

    Classe DTO contendo as seguintes propriedades:
    - answer: bool
    - explanation: str
    - analyzed_document: dict
    """

    answer: bool = Field(
        description="""Propriedade <answer> que deve conter a resposta podendo ser True
        ou False"""
    )
    explanation: str = Field(
        description="""Propriedade <explanation> que deve conter a explicação, com no
        máximo 100 caracteres, sucinta do score"""
    )
    analyzed_document: dict = Field(
        description="""Propriedade <dict> que deve conter o documento
        analisado"""
    )


class GraphState(BaseModel):
    """Classe que representa o estado na nossa aplicação (Graph)."""

    question: str = Field(default=None)
    response: str = Field(default=None)
    documents: list[Document] = Field(default_factory=list)
    codes: list[Document] = Field(default_factory=list)
    history: list[BaseMessage] = Field(default_factory=list)
